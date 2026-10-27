//! reward.rs — a scalar reward model trained on a contrastive pair set.
//!
//! Every other trainer in Ster produces a model that writes text. This one
//! produces a model that *judges* it: a single scalar per sequence, higher for
//! the response an operator preferred. It exists because policy optimization
//! needs a reward, and a reward that is a language model's own likelihood is
//! circular.
//!
//! The objective is Bradley-Terry, which says the probability that the chosen
//! response beats the rejected one is `sigmoid(r_chosen - r_rejected)`. Two
//! consequences fall straight out of that difference and shape the code:
//!
//! * **The head has no bias.** A bias is added to both scores and cancels in
//!   the difference, so it would be a parameter with an identically zero
//!   gradient — dead weight in every sense.
//!
//! * **Only differences are learned.** The absolute scale and offset of the
//!   scores are not identified by the objective, so the numbers in the report
//!   are meaningful against each other and not against any external unit. The
//!   head is zero-initialised, which pins the starting point at "no opinion":
//!   every sequence scores exactly zero, and the first loss is therefore
//!   exactly `ln 2`, the same identity check the preference losses give. That
//!   starting point is also why the report counts `tied_pairs`: a tie is not a
//!   strict win, so a head that has not moved reports `accuracy` 0.0, which
//!   reads exactly like a head that ranked every pair backwards. The tie count
//!   separates them without redefining the accuracy anyone already consumes.
//!
//! The head trains together with the adapters beneath it, in one `VarMap` and
//! under one optimizer, and is written into the same safetensors file. A head
//! is a function of the residual stream the adapters shaped; pairing one with
//! adapters it never saw would produce scores that mean nothing, so the
//! artifact does not offer that as a possibility. The head is in that map, so
//! a checkpoint carries it and its moments with the adapters.

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{ParamsAdamW, VarMap};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use super::super::{
    batch,
    checkpoint::{Position, optimizer::AdamW},
    preflight::{Preflight, Trainable, encode_pairs, pair_set_label},
    schedule,
};
use crate::{artifact::PairSet, runtime::Runtime, workflow};

mod head;
mod options;
mod step;

pub use head::{RewardHead, RewardModel};
pub use options::{RewardOptions, RewardReport};

use options::{Running, Totals};
use step::{Summary, step_loss};

/// Trains `head` and the adapters `varmap` owns to rank each pair.
///
/// `runtime`, `varmap` and `head` must be the three halves of one model:
/// `Runtime::load_trainable` returns the first two, and `RewardHead::fresh`
/// registers the third in that same map. A head registered elsewhere would
/// score correctly and never be stepped.
pub fn reward(
    runtime: &Runtime,
    varmap: &VarMap,
    head: &RewardHead,
    pairs: &PairSet,
    options: &RewardOptions,
) -> Result<RewardReport> {
    pairs.validate(&pair_set_label(pairs))?;
    let Trainable { spec, vars: _, tensors, parameters, limit } = Preflight {
        subject: "reward modeling",
        unit: "pair",
        pass: "epoch",
        // The head is in this map too, so the count is not adapters alone.
        noun: "tensors",
        epochs: options.epochs,
        accumulation: options.accumulation,
        batch: options.batch,
        learning_rate: options.learning_rate,
        max_sequence: options.max_sequence,
    }
    .open(runtime, varmap, &options.spec)?;

    let encoded = encode_pairs(runtime, pairs, limit)?;

    let fingerprint = options.fingerprint();
    let mut optimizer = AdamW::over(varmap, ParamsAdamW { lr: options.learning_rate, ..Default::default() })
        .context("failed to initialize the AdamW optimizer")?;
    let resumed = options.durable.load(&mut optimizer, &fingerprint)?;
    let mut running: Running = match &resumed {
        Some(position) => serde_json::from_value(position.running.clone())
            .context("the checkpoint's running report is not one reward modeling wrote")?,
        None => Running::default(),
    };
    let mut checkpoints = Vec::new();

    // Both sides of a pair go through the same forward, so the width its rows
    // are padded to is the longer of the two.
    let lengths: Vec<usize> = encoded.iter().map(|pair| pair.chosen.len().max(pair.rejected.len())).collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let total_steps = batch::steps_per_epoch(encoded.len(), options.batch, options.accumulation) * options.epochs;
    let mut order: Vec<usize> = encoded.iter().enumerate().map(|(slot, _)| slot).collect();

    for (epoch, _) in std::iter::repeat_n((), options.epochs).enumerate() {
        // Reseeded per epoch; a resumed run replays the shuffles of the
        // epochs it skips.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let plans = batch::plan(&order, &lengths, options.batch, options.accumulation);
        let remaining = match &resumed {
            Some(position) if position.pass > epoch => continue,
            Some(position) if position.pass == epoch => plans.get(position.within..).with_context(|| {
                format!("the checkpoint stands {} steps into epoch {epoch}, which has {} steps", position.within, plans.len())
            })?,
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
            // `batch` pairs per forward — the chosen side then the rejected
            // one, right-padded to a common width and masked so neither side
            // reads the other's filler — and `accumulation` forwards per
            // optimizer step. The head still reads one row's last real
            // position, because `batch::row` slices each row back to its own
            // length before the head sees it. Each pair's loss is divided by
            // the constant `accumulation * batch`, so a short tail steps
            // proportionally smaller.
            let mut summed: Option<Tensor> = None;
            let mut losses: Vec<f64> = Vec::with_capacity(plan.units);
            for forward in &plan.forwards {
                let rows: Vec<&[u32]> = forward
                    .iter()
                    .flat_map(|&slot| [encoded[slot].chosen.as_slice(), encoded[slot].rejected.as_slice()])
                    .collect();
                let sides = rows.len() / forward.len();
                let read = batch::read_rows(&rows, options.batch, sides, |pass| runtime.forward_hidden_rows(pass))?;
                for (pair_rows, &slot) in read.chunks_exact(sides).zip(forward) {
                    let pair = &encoded[slot];
                    let [chosen, rejected] = pair_rows else {
                        bail!("pair {} was read as {} rows, not its two sides", pair.index, pair_rows.len());
                    };
                    let value = step_loss(head, chosen, rejected)
                        .with_context(|| format!("pair {} produced no usable loss", pair.index))?;
                    losses.push(value.loss);
                    summary.record(&value);
                    let scaled = (value.tensor / scale)?;
                    summed = Some(match summed {
                        Some(total) => (total + scaled)?,
                        None => scaled,
                    });
                }
            }
            let Some(summed) = summed else {
                // `plan` never yields a step with no forwards and never a
                // forward with no rows; refusing beats stepping on nothing.
                bail!("an accumulation group contained no pairs");
            };
            optimizer.backward_step(&summed).context("failed to backpropagate the accumulated loss")?;

            let group_mean = (losses.iter().sum::<f64>() / plan.units as f64) as f32;
            running.first_loss.get_or_insert(group_mean);
            running.final_loss = Some(group_mean);
            running.steps.push(group_mean);
            running.epoch = Totals::from(&summary);
            let step = optimizer.rates.len();
            workflow::progress(format!(
                "epochs done {}/{} step {step}/{total_steps} pair {}/{} loss {group_mean:.4} accuracy {:.3}",
                running.finished.len(),
                options.epochs,
                summary.pairs,
                encoded.len(),
                summary.accuracy()
            ));
            if options.durable.due(step) {
                let position =
                    Position { pass: epoch, step, within: running.steps.len(), running: serde_json::to_value(&running)? };
                options.durable.save(&optimizer, &fingerprint, &position)?;
                checkpoints.push(step);
            }
        }

        running.finished.push(Totals::from(&summary));
        workflow::progress(format!(
            "epochs done {}/{} mean loss {:.4} accuracy {:.3} ties {}/{} score margin {:.4}",
            running.finished.len(),
            options.epochs,
            summary.mean_loss(),
            summary.accuracy(),
            summary.tied,
            summary.pairs,
            summary.mean_margin()
        ));
    }

    let (Some(first_loss), Some(final_loss), Some(last)) =
        (running.first_loss, running.final_loss, running.finished.last())
    else {
        bail!("the run took no step, so it has no loss to report");
    };
    let epoch_summary = Summary::from(last);
    Ok(RewardReport {
        pairs: pairs.pairs.len(),
        trained_pairs: encoded.len(),
        skipped_long: pairs.pairs.len() - encoded.len(),
        epochs: options.epochs,
        steps: optimizer.rates.len(),
        trainable_tensors: tensors,
        trainable_parameters: parameters,
        first_loss,
        final_loss,
        mean_final_epoch_loss: epoch_summary.mean_loss(),
        accuracy: epoch_summary.accuracy(),
        tied_pairs: epoch_summary.tied,
        mean_chosen_score: epoch_summary.mean_chosen(),
        mean_rejected_score: epoch_summary.mean_rejected(),
        mean_score_margin: epoch_summary.mean_margin(),
        rank: spec.rank,
        alpha: spec.alpha,
        targets: spec.targets.iter().map(|target| target.name().to_owned()).collect(),
        layers: spec.layers.clone(),
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        batch: options.batch,
        resumed_from: resumed.map(|position| position.step),
        checkpoints,
    })
}
