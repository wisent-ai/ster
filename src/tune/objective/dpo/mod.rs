//! dpo.rs — direct preference optimization over a contrastive pair set.
//!
//! Supervised fine-tuning can only say "produce this". A preference set says
//! "prefer this over that", which is a different statement and needs a
//! different objective: raise the policy's log-probability of the chosen side
//! and lower it on the rejected side, both measured *relative to a frozen
//! reference* so the policy cannot win by becoming more confident about
//! everything at once.
//!
//! Three things about this implementation are decisions rather than details.
//!
//! * **The reference is this same model with the adapters switched off.**
//!   `Runtime::load_trainable` maps the base weights read-only and registers
//!   only the low-rank pairs, and `B` starts at zero, so the base *is* the
//!   model the policy was initialised as. [`Route::Base`] skips the update at
//!   every projection and reproduces it exactly. A second copy of the
//!   checkpoint would double the resident set to compute numbers these weights
//!   already hold.
//!
//! * **The reference is scored once for the whole run.** A frozen model cannot
//!   move, so its log-probability of a fixed sequence is a constant. Computing
//!   it in one pass up front costs a forward per side of every pair instead of
//!   one per side per epoch, and it means a run that is going to fail on a
//!   too-long pair fails before the first gradient. A resumed run scores it
//!   again, which gives the same constants.
//!
//! * **A pair is scored whole, not split into prompt and completion.** A
//!   [`PairSet`] carries two complete texts and no prompt field, and it does
//!   not need one: when the two sides share a leading prefix — which is exactly
//!   what `ster pairs synthesize` writes, `Question: …\nAnswer: …` — that
//!   prefix sits at the same positions in both sequences, so its
//!   log-probability is the same expression on both sides of the margin and
//!   cancels out of the value *and* the gradient.
//!
//! A run given a checkpoint directory writes its state every
//! `--checkpoint-every` steps and `--resume` continues it exactly, as
//! supervised fine-tuning does ([`super::super::checkpoint`]).

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

mod bipo;
mod options;
mod scoring;
mod strength;

pub use bipo::{BipoOptions, BipoReport, bipo};
pub use options::{DpoLoss, DpoOptions, DpoReport};
pub use strength::{StrengthCandidate, StrengthOptions, StrengthReport, strengths};

use options::{Running, Totals};
use scoring::{Preference, Scored, Summary, reference_scores, step_loss};

/// Trains the adapters `varmap` owns to prefer each pair's positive side.
///
/// `runtime` must be the one `Runtime::load_trainable` returned alongside
/// `varmap`; the two are a pair, and passing a plain `Runtime` here would
/// produce a run whose optimizer has no variables to step and whose reference
/// pass is indistinguishable from its policy pass.
pub fn dpo(
    runtime: &Runtime,
    varmap: &VarMap,
    pairs: &PairSet,
    options: &DpoOptions,
) -> Result<DpoReport> {
    // DPO's loss is `-log σ(β · margin)`; at β = 0 every pair's loss is the
    // constant `log 2` and no gradient reaches the policy, so β must be a
    // finite number above zero (Rafailov et al., eq. 7):
    // https://arxiv.org/abs/2305.18290
    if !(options.beta.is_finite() && options.beta > 0.0) {
        bail!("direct preference optimization requires a finite beta above zero");
    }
    pairs.validate(&pair_set_label(pairs))?;
    let Trainable {
        spec,
        vars: _,
        tensors,
        parameters,
        limit,
    } = Preflight {
        subject: "direct preference optimization",
        unit: "pair",
        pass: "epoch",
        noun: "adapter tensors",
        epochs: options.epochs,
        accumulation: options.accumulation,
        batch: Some(options.batch),
        learning_rate: options.learning_rate,
        max_sequence: options.max_sequence,
    }
    .open(runtime, varmap, &options.spec)?;

    let mut encoded: Vec<Scored> = encode_pairs(runtime, pairs, limit)?
        .into_iter()
        .map(Scored::new)
        .collect();
    reference_scores(runtime, &mut encoded, options.batch)?;

    let fingerprint = options.fingerprint();
    let mut optimizer = AdamW::over(
        varmap,
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

    // A pair's two sides go through the same forward, so the width its rows
    // are padded to is the longer of the two; grouping on that is grouping on
    // what the padding actually costs.
    let lengths: Vec<usize> = encoded
        .iter()
        .map(|scored| scored.pair.chosen.len().max(scored.pair.rejected.len()))
        .collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let total_steps =
        batch::steps_per_epoch(encoded.len(), options.batch, options.accumulation) * options.epochs;
    let mut order: Vec<usize> = encoded.iter().enumerate().map(|(slot, _)| slot).collect();
    let preference = Preference {
        loss: options.loss,
        beta: options.beta,
    };

    for (epoch, _) in std::iter::repeat_n((), options.epochs).enumerate() {
        // Reseeded per epoch, exactly as supervised fine-tuning is; a resumed
        // run replays the shuffles of the epochs it skips.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let plans = batch::plan(&order, &lengths, options.batch, options.accumulation);
        let remaining = match &resumed {
            Some(position) if position.pass > epoch => continue,
            Some(position) if position.pass == epoch => {
                plans.get(position.within..).with_context(|| {
                    format!(
                        "the checkpoint stands {} steps into epoch {epoch}, which has {} steps",
                        position.within,
                        plans.len()
                    )
                })?
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
            // `batch` pairs per forward — the chosen side then the rejected
            // one, right-padded to a common width and masked so neither side
            // reads the other's filler — and `accumulation` forwards per
            // optimizer step. Each pair's loss is divided by the constant
            // `accumulation * batch`, so a short tail steps proportionally
            // smaller rather than as far as a full step.
            let mut summed: Option<Tensor> = None;
            let mut losses: Vec<f64> = Vec::with_capacity(plan.units);
            for forward in &plan.forwards {
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
                    runtime.forward_train_rows(pass)
                })?;
                for (pair_rows, &slot) in read.chunks_exact(sides).zip(forward) {
                    let scored = &encoded[slot];
                    let [chosen, rejected] = pair_rows else {
                        bail!(
                            "pair {} was read as {} rows, not its two sides",
                            scored.pair.index,
                            pair_rows.len()
                        );
                    };
                    let value = step_loss(runtime, scored, chosen, rejected, preference)
                        .with_context(|| {
                            format!("pair {} produced no usable loss", scored.pair.index)
                        })?;
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
            optimizer
                .backward_step(&summed)
                .context("failed to backpropagate the accumulated loss")?;

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
        workflow::progress(format!(
            "epochs done {}/{} mean loss {:.4} accuracy {:.3} reward margin {:.4}",
            running.finished.len(),
            options.epochs,
            summary.mean_loss(),
            summary.accuracy(),
            summary.mean_margin()
        ));
    }

    let (Some(first_loss), Some(final_loss), Some(last)) = (
        running.first_loss,
        running.final_loss,
        running.finished.last(),
    ) else {
        bail!("the run took no step, so it has no loss to report");
    };
    let epoch_summary = Summary::from(last);
    Ok(DpoReport {
        loss: options.loss.name().to_owned(),
        beta: options.beta,
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
        mean_reward_margin: epoch_summary.mean_margin(),
        mean_chosen_reward: epoch_summary.mean_chosen(),
        mean_rejected_reward: epoch_summary.mean_rejected(),
        rank: spec.rank,
        alpha: spec.alpha,
        targets: spec
            .targets
            .iter()
            .map(|target| target.name().to_owned())
            .collect(),
        layers: spec.layers.clone(),
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        batch: options.batch,
        resumed_from: resumed.map(|position| position.step),
        checkpoints,
    })
}
