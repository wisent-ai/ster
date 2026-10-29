//! bipo — bi-directional preference optimization of one steering vector.
//!
//! BiPO (Cao et al., "Personalized Steering of Large Language Models via
//! Bi-directional Preference Optimization", arXiv 2406.00045) learns the
//! vector itself instead of reading it off activations: the policy is the
//! frozen model with `d * strength * v` added to the residual stream at one
//! layer, and the loss is DPO's (or IPO's) against the same model with
//! nothing added. Each forward draws its direction `d` from {+, -}: with `+v`
//! the policy is pushed to prefer each pair's chosen side, with `-v` its
//! rejected side — the same loss with beta negated — so one vector learns to
//! move the behaviour both ways, which is what lets `ster generate --strength`
//! turn it up, down, or past zero afterwards.
//!
//! What is shared with `dpo` is shared on purpose: pairs are encoded and
//! skipped the same way, the reference is scored once up front with nothing
//! added, forwards are planned and batched by the same planner, each pair's
//! loss is `step_loss`, and the running summary and what a checkpoint carries
//! of it are the same. What differs is what moves. No adapter is registered
//! and no base weight is trainable; the one variable is `v`, which starts at
//! zero so the first policy is the reference itself.
//!
//! A run given a checkpoint directory writes the vector, its optimizer
//! moments and where the run stands every `--checkpoint-every` steps, and a
//! run given `--resume` continues from the newest one exactly where it
//! stopped: it replays the shuffle and the direction draws of every step
//! already taken, so the steps after it draw the directions the stopped run
//! would have drawn.

mod options;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Tensor, Var};
use candle_nn::{ParamsAdamW, VarMap};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

pub use options::{BipoOptions, BipoReport};

use super::super::super::{
    batch,
    checkpoint::{Position, optimizer::AdamW},
    preflight::{encode_pairs, pair_set_label},
    schedule,
};
use super::options::{Running, Totals};
use super::scoring::{Preference, Scored, Summary, reference_scores, step_loss};
use crate::{
    artifact::{LayerVector, PairSet, SteeringArtifact},
    model::SteeringPlan,
    runtime::Runtime,
    workflow,
};

/// The name the steering vector is kept under in the optimizer and in a
/// checkpoint.
const VECTOR: &str = "steering";

/// Learns the steering vector at `options.layer` that makes the model prefer
/// each pair's chosen side when added and its rejected side when subtracted,
/// and returns it as a steering artifact with the run's report.
pub fn bipo(
    runtime: &Runtime,
    pairs: &PairSet,
    options: &BipoOptions,
) -> Result<(SteeringArtifact, BipoReport)> {
    options.validate(runtime.layer_count())?;
    pairs.validate(&pair_set_label(pairs))?;
    let mut encoded: Vec<Scored> = encode_pairs(runtime, pairs, options.max_sequence)?
        .into_iter()
        .map(Scored::new)
        .collect();
    let skipped_long = pairs.pairs.len() - encoded.len();
    reference_scores(runtime, &mut encoded, options.batch)?;

    let hidden = runtime.hidden_size();
    let vector = Var::zeros(hidden, DType::F32, runtime.device())
        .context("failed to create the steering vector")?;
    let varmap = VarMap::new();
    varmap
        .data()
        .lock()
        .map_err(|poisoned| anyhow!("the run's tensor map is unreadable: {poisoned}"))?
        .insert(VECTOR.to_owned(), vector.clone());
    let fingerprint = options.fingerprint();
    let mut optimizer = AdamW::over(
        &varmap,
        ParamsAdamW {
            lr: options.learning_rate,
            ..Default::default()
        },
    )
    .context("failed to initialize the AdamW optimizer")?;
    let resumed = options.durable.load(&mut optimizer, &fingerprint)?;
    let mut running: Running = match &resumed {
        Some(position) => serde_json::from_value(position.running.clone())
            .context("the checkpoint's running report is not one preference optimization wrote")?,
        None => Running::default(),
    };
    let mut checkpoints = Vec::new();

    let lengths: Vec<usize> = encoded
        .iter()
        .map(|scored| scored.pair.chosen.len().max(scored.pair.rejected.len()))
        .collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let total_steps =
        batch::steps_per_epoch(encoded.len(), options.batch, options.accumulation) * options.epochs;
    let mut order: Vec<usize> = encoded.iter().enumerate().map(|(slot, _)| slot).collect();

    for (epoch, _) in std::iter::repeat_n((), options.epochs).enumerate() {
        // Reseeded per epoch, as dpo is, so a run is reproducible from `seed`
        // alone; the same generator draws each forward's direction.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let plans = batch::plan(&order, &lengths, options.batch, options.accumulation);
        let remaining = match &resumed {
            Some(position) if position.pass > epoch => continue,
            Some(position) if position.pass == epoch => {
                let (taken, remaining) =
                    plans.split_at_checked(position.within).with_context(|| {
                        format!(
                            "the checkpoint stands {} steps into epoch {epoch}, which has {} steps",
                            position.within,
                            plans.len()
                        )
                    })?;
                // One direction was drawn per forward of every step taken;
                // drawing them again leaves the generator where it stood.
                for _ in taken.iter().flat_map(|plan| &plan.forwards) {
                    rng.random::<bool>();
                }
                remaining
            }
            _ => {
                running.steps.clear();
                running.epoch = Totals::default();
                &plans[..]
            }
        };
        let mut summary = Summary::from(&running.epoch);

        for plan in remaining {
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                optimizer.rates.len(),
                total_steps,
                options.warmup_steps,
            ));
            let mut summed: Option<Tensor> = None;
            let mut pair_losses: Vec<f64> = Vec::with_capacity(plan.units);
            for forward in &plan.forwards {
                let subtract: bool = rng.random();
                let (strength, beta) = if subtract {
                    (-options.strength, -options.beta)
                } else {
                    (options.strength, options.beta)
                };
                let steering = SteeringPlan::from_tensors(
                    [(options.layer, vector.as_tensor().clone())],
                    strength,
                    hidden,
                    runtime.dtype(),
                )?;
                let rows: Vec<&[u32]> = forward
                    .iter()
                    .flat_map(|&slot| {
                        [
                            encoded[slot].pair.chosen.as_slice(),
                            encoded[slot].pair.rejected.as_slice(),
                        ]
                    })
                    .collect();
                let sides = rows.len() / forward.len();
                let read = batch::read_rows(&rows, options.batch, sides, |pass| {
                    runtime.forward_steered_rows(pass, &steering)
                })?;
                for (&slot, logits) in forward.iter().zip(read.chunks_exact(sides)) {
                    let [chosen, rejected] = logits else {
                        bail!(
                            "a preference pair read back {} rows instead of a chosen and a rejected side",
                            logits.len()
                        );
                    };
                    let scored = &encoded[slot];
                    let value = step_loss(
                        runtime,
                        scored,
                        chosen,
                        rejected,
                        Preference {
                            loss: options.loss,
                            beta,
                        },
                    )
                    .with_context(|| {
                        format!("pair {} produced no usable loss", scored.pair.index)
                    })?;
                    pair_losses.push(value.loss);
                    summary.record(&value);
                    let scaled = (value.tensor / scale)?;
                    summed = Some(match summed {
                        Some(total) => (total + scaled)?,
                        None => scaled,
                    });
                }
            }
            let Some(summed) = summed else {
                bail!("an accumulation group contained no pairs");
            };
            optimizer
                .backward_step(&summed)
                .context("failed to backpropagate the accumulated loss")?;
            let mean = (pair_losses.iter().sum::<f64>() / plan.units as f64) as f32;
            running.first_loss.get_or_insert(mean);
            running.final_loss = Some(mean);
            running.steps.push(mean);
            running.epoch = Totals::from(&summary);
            let step = optimizer.rates.len();
            workflow::progress(format!(
                "epochs done {}/{} step {step}/{total_steps} loss {mean:.4} accuracy {:.3}",
                running.finished.len(),
                options.epochs,
                summary.accuracy()
            ));
            if options.durable.due(step) {
                let position = Position {
                    pass: epoch,
                    step,
                    within: running.steps.len(),
                    running: serde_json::to_value(&running)?,
                };
                options.durable.save(&optimizer, &fingerprint, &position)?;
                checkpoints.push(step);
            }
        }
        running.finished.push(Totals::from(&summary));
    }

    let (Some(first_loss), Some(final_loss), Some(last)) = (
        running.first_loss,
        running.final_loss,
        running.finished.last(),
    ) else {
        bail!("the run took no step, so it has no loss to report");
    };
    let epoch_summary = Summary::from(last);
    let values: Vec<f32> = vector.as_tensor().to_vec1::<f32>()?;
    let vector_norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let report = BipoReport {
        layer: options.layer,
        strength: options.strength,
        loss: options.loss.name().to_owned(),
        beta: options.beta,
        pairs: pairs.pairs.len(),
        trained_pairs: encoded.len(),
        skipped_long,
        epochs: options.epochs,
        steps: optimizer.rates.len(),
        first_loss,
        final_loss,
        mean_final_epoch_loss: epoch_summary.mean_loss(),
        accuracy: epoch_summary.accuracy(),
        mean_reward_margin: epoch_summary.mean_margin(),
        vector_norm,
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        batch: options.batch,
        resumed_from: resumed.map(|position| position.step),
        checkpoints,
    };
    let artifact = SteeringArtifact::new(
        runtime.model_id.clone(),
        runtime.revision.clone(),
        pairs.trait_name.clone(),
        "bipo".to_owned(),
        hidden,
        vec![LayerVector {
            layer: options.layer,
            values,
            train_margin: report.mean_reward_margin,
            train_accuracy: report.accuracy,
        }],
        runtime.precision(),
        runtime.chat_status(),
    );
    Ok((artifact, report))
}
