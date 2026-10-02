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
//! artifact does not offer that as a possibility.

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use super::super::{
    batch,
    preflight::{Preflight, Trainable, encode_pairs, pair_set_label},
    schedule,
};
use crate::{artifact::PairSet, lora, runtime::Runtime, workflow};

mod head;
mod step;

pub use head::{RewardHead, RewardModel};

use step::{Summary, step_loss};

#[derive(Debug, Clone)]
pub struct RewardOptions {
    pub spec: lora::Spec,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
    /// Pairs folded into one forward pass. A pair is two rows, so a batch of
    /// four pairs is a forward of eight sequences; one is the unbatched pass
    /// every run recorded before batching existed.
    pub batch: usize,
    pub warmup_steps: usize,
    pub max_sequence: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RewardReport {
    pub pairs: usize,
    pub trained_pairs: usize,
    pub skipped_long: usize,
    pub epochs: usize,
    pub steps: usize,
    pub trainable_tensors: usize,
    pub trainable_parameters: usize,
    pub first_loss: f32,
    pub final_loss: f32,
    pub mean_final_epoch_loss: f32,
    /// The share of pairs the head already ranks correctly, over the final
    /// epoch — so it describes the head that was written, not an average over
    /// a head that was still moving.
    pub accuracy: f32,
    /// How many of those pairs the head scored exactly equal, over the same
    /// final epoch — the pairs `accuracy` counts as not-correct without them
    /// being wrong. A fresh head is zeros, so a run short enough that it never
    /// moved reports every pair tied here and 0.0 there; a head that learned
    /// the order backwards reports the same accuracy with no ties at all.
    pub tied_pairs: usize,
    pub mean_chosen_score: f32,
    pub mean_rejected_score: f32,
    pub mean_score_margin: f32,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub layers: Vec<usize>,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
}

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
    let Trainable {
        spec,
        vars,
        tensors,
        parameters,
        limit,
    } = Preflight {
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
    let skipped_long = pairs.pairs.len() - encoded.len();

    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr: options.learning_rate,
            ..Default::default()
        },
    )
    .context("failed to initialize the AdamW optimizer")?;

    // Both sides of a pair go through the same forward, so the width its rows
    // are padded to is the longer of the two.
    let lengths: Vec<usize> = encoded
        .iter()
        .map(|pair| pair.chosen.len().max(pair.rejected.len()))
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
            // step. The head still reads one row's last real position, because
            // `batch::row` slices each row back to its own length before the
            // head sees it. Each pair's loss is divided by the constant
            // `accumulation * batch`, so a short tail steps proportionally
            // smaller.
            let mut summed: Option<Tensor> = None;
            let mut group_loss = 0f64;
            for forward in &plan.forwards {
                let mut rows: Vec<&[u32]> = Vec::with_capacity(forward.len() * 2);
                for &slot in forward {
                    rows.push(&encoded[slot].chosen);
                    rows.push(&encoded[slot].rejected);
                }
                let read = batch::read_rows(&rows, options.batch, 2, |pass| {
                    runtime.forward_hidden_rows(pass)
                })?;
                for (position, &slot) in forward.iter().enumerate() {
                    let pair = &encoded[slot];
                    let value = step_loss(head, &read[position * 2], &read[position * 2 + 1])
                        .with_context(|| format!("pair {} produced no usable loss", pair.index))?;
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
            "epoch {}/{} mean loss {:.4} accuracy {:.3} ties {}/{} score margin {:.4}",
            epoch + 1,
            options.epochs,
            summary.mean_loss(),
            summary.accuracy(),
            summary.tied,
            summary.pairs,
            summary.mean_margin()
        ));
        epoch_summary = summary;
    }

    Ok(RewardReport {
        pairs: pairs.pairs.len(),
        trained_pairs: encoded.len(),
        skipped_long,
        epochs: options.epochs,
        steps: step,
        trainable_tensors: tensors,
        trainable_parameters: parameters,
        // At least one group ran: `encode_pairs` refuses an empty result and
        // `epochs` is at least one. The fallback keeps the report
        // serializable rather than emitting a JSON null for a float field.
        first_loss: first_loss.unwrap_or(final_loss),
        final_loss,
        mean_final_epoch_loss: epoch_summary.mean_loss(),
        accuracy: epoch_summary.accuracy(),
        tied_pairs: epoch_summary.tied,
        mean_chosen_score: epoch_summary.mean_chosen(),
        mean_rejected_score: epoch_summary.mean_rejected(),
        mean_score_margin: epoch_summary.mean_margin(),
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
