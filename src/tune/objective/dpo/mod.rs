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
//!   it in one pass up front costs `2 * pairs` forwards instead of
//!   `2 * pairs * epochs`, and it means a run that is going to fail on a
//!   too-long pair fails before the first gradient.
//!
//! * **A pair is scored whole, not split into prompt and completion.** A
//!   [`PairSet`] carries two complete texts and no prompt field, and it does
//!   not need one: when the two sides share a leading prefix — which is exactly
//!   what `ster pairs synthesize` writes, `Question: …\nAnswer: …` — that
//!   prefix sits at the same positions in both sequences, so its
//!   log-probability is the same expression on both sides of the margin and
//!   cancels out of the value *and* the gradient. Scoring the whole text is
//!   therefore not an approximation of a prompt-conditioned objective; on a
//!   prefix-sharing set it is the same objective.

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use super::super::{
    batch,
    preflight::{Preflight, Trainable, encode_pairs, pair_set_label},
    schedule,
};
use crate::{artifact::PairSet, runtime::Runtime, workflow};

mod bipo;
mod options;
mod scoring;

pub use bipo::{BipoOptions, BipoReport, bipo};
pub use options::{DpoLoss, DpoOptions, DpoReport};

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
    if !options.beta.is_finite() || options.beta <= 0.0 {
        bail!("direct preference optimization requires a finite beta above zero");
    }
    pairs.validate(&pair_set_label(pairs))?;
    let Trainable {
        spec,
        vars,
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
        batch: options.batch,
        learning_rate: options.learning_rate,
        max_sequence: options.max_sequence,
    }
    .open(runtime, varmap, &options.spec)?;

    let mut encoded: Vec<Scored> = encode_pairs(runtime, pairs, limit)?
        .into_iter()
        .map(Scored::new)
        .collect();
    let skipped_long = pairs.pairs.len() - encoded.len();
    reference_scores(runtime, &mut encoded, options.batch)?;

    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr: options.learning_rate,
            ..Default::default()
        },
    )
    .context("failed to initialize the AdamW optimizer")?;

    // A pair's two sides go through the same forward, so the width its rows
    // are padded to is the longer of the two; grouping on that is grouping on
    // what the padding actually costs.
    let lengths: Vec<usize> = encoded
        .iter()
        .map(|scored| scored.pair.chosen.len().max(scored.pair.rejected.len()))
        .collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let steps_per_epoch =
        batch::steps_per_epoch(encoded.len(), options.batch, options.accumulation);
    let total_steps = steps_per_epoch * options.epochs;
    let mut order: Vec<usize> = (0..encoded.len()).collect();
    let mut step = 0usize;
    let mut first_loss: Option<f32> = None;
    let mut final_loss = 0f32;
    let mut epoch_summary = Summary::default();

    for epoch in 0..options.epochs {
        // Reseeded per epoch rather than carried across epochs so that a run is
        // reproducible from `seed` alone, exactly as supervised fine-tuning is.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let mut summary = Summary::default();

        for plan in batch::plan(&order, &lengths, options.batch, options.accumulation) {
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                step,
                total_steps,
                options.warmup_steps,
            ));

            // `batch` pairs per forward — two rows each, chosen then rejected,
            // right-padded to a common width and masked so neither side reads
            // the other's filler — and `accumulation` forwards per optimizer
            // step. Each pair's loss is divided by the constant
            // `accumulation * batch`, so a short tail steps proportionally
            // smaller rather than as far as a full step.
            let mut summed: Option<Tensor> = None;
            let mut group_loss = 0f64;
            for forward in &plan.forwards {
                let mut rows: Vec<&[u32]> = Vec::with_capacity(forward.len() * 2);
                for &slot in forward {
                    rows.push(&encoded[slot].pair.chosen);
                    rows.push(&encoded[slot].pair.rejected);
                }
                let read = batch::read_rows(&rows, options.batch, 2, |pass| {
                    runtime.forward_train_rows(pass)
                })?;
                for (position, &slot) in forward.iter().enumerate() {
                    let scored = &encoded[slot];
                    let chosen = &read[position * 2];
                    let rejected = &read[position * 2 + 1];
                    let preference = Preference {
                        loss: options.loss,
                        beta: options.beta,
                    };
                    let value = step_loss(runtime, scored, chosen, rejected, preference)
                        .with_context(|| {
                            format!("pair {} produced no usable loss", scored.pair.index)
                        })?;
                    group_loss += value.loss;
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

            let group_mean = (group_loss / plan.units as f64) as f32;
            step += 1;
            if first_loss.is_none() {
                first_loss = Some(group_mean);
            }
            final_loss = group_mean;
            workflow::progress(format!(
                "epoch {}/{} step {step}/{total_steps} pair {}/{} loss {group_mean:.4} accuracy {:.3}",
                epoch + 1,
                options.epochs,
                summary.pairs,
                encoded.len(),
                summary.accuracy()
            ));
        }

        workflow::progress(format!(
            "epoch {}/{} mean loss {:.4} accuracy {:.3} reward margin {:.4}",
            epoch + 1,
            options.epochs,
            summary.mean_loss(),
            summary.accuracy(),
            summary.mean_margin()
        ));
        epoch_summary = summary;
    }

    Ok(DpoReport {
        loss: options.loss.name().to_owned(),
        beta: options.beta,
        pairs: pairs.pairs.len(),
        trained_pairs: encoded.len(),
        skipped_long,
        epochs: options.epochs,
        steps: step,
        trainable_tensors: tensors,
        trainable_parameters: parameters,
        // At least one group ran: `encoded` is non-empty and `epochs` is at
        // least one. The fallback keeps the report serializable rather than
        // emitting a JSON null for a float field.
        first_loss: first_loss.unwrap_or(final_loss),
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
    })
}
