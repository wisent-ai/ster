//! bipo.rs — bi-directional preference optimization of one steering vector.
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
//! loss is `step_loss`, and the running summary is the same. What differs is
//! what moves. No adapter is registered and no base weight is trainable; the
//! one variable is `v`, which starts at zero so the first policy is the
//! reference itself.

use std::num::NonZeroUsize;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use super::super::super::{
    batch,
    preflight::{encode_pairs, pair_set_label},
    schedule,
};
use super::DpoLoss;
use super::scoring::{Preference, Scored, Summary, reference_scores, step_loss};
use crate::{
    artifact::{LayerVector, PairSet, SteeringArtifact},
    model::SteeringPlan,
    runtime::Runtime,
    workflow,
};

/// Every setting one BiPO run takes; Ster assumes none of them.
#[derive(Debug, Clone)]
pub struct BipoOptions {
    /// The layer whose residual stream the vector is added to.
    pub layer: usize,
    /// The multiple of `v` added while training, in each direction.
    pub strength: f64,
    pub loss: DpoLoss,
    pub beta: f64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
    pub warmup_steps: usize,
    pub max_sequence: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BipoReport {
    pub layer: usize,
    pub strength: f64,
    pub loss: String,
    pub beta: f64,
    pub pairs: usize,
    pub trained_pairs: usize,
    pub skipped_long: usize,
    pub epochs: usize,
    pub steps: usize,
    pub first_loss: Option<f32>,
    pub final_loss: Option<f32>,
    pub mean_final_epoch_loss: f32,
    pub accuracy: f32,
    pub mean_reward_margin: f32,
    pub vector_norm: f32,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
}

/// Refuses a run whose settings cannot train: each refusal names its setting.
fn validate(options: &BipoOptions, layers: usize) -> Result<()> {
    if options.layer >= layers {
        bail!(
            "layer {} is outside the model's {layers} layers",
            options.layer
        );
    }
    for (name, value) in [
        ("beta", options.beta),
        ("strength", options.strength),
        ("learning rate", options.learning_rate),
    ] {
        // A normal, positive float: finite, not zero, not subnormal.
        if !(value.is_normal() && value.is_sign_positive()) {
            bail!(
                "bi-directional preference optimization requires a finite {name} above zero, not {value}"
            );
        }
    }
    for (name, value) in [
        ("epochs", options.epochs),
        ("accumulation", options.accumulation),
        ("batch size", options.batch),
        ("sequence limit", options.max_sequence),
    ] {
        if NonZeroUsize::new(value).is_none() {
            bail!("bi-directional preference optimization requires {name} of at least one");
        }
    }
    Ok(())
}

/// Learns the steering vector at `options.layer` that makes the model prefer
/// each pair's chosen side when added and its rejected side when subtracted,
/// and returns it as a steering artifact with the run's report.
pub fn bipo(
    runtime: &Runtime,
    pairs: &PairSet,
    options: &BipoOptions,
) -> Result<(SteeringArtifact, BipoReport)> {
    validate(options, runtime.layer_count())?;
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
    let mut optimizer = AdamW::new(
        vec![vector.clone()],
        ParamsAdamW {
            lr: options.learning_rate,
            ..Default::default()
        },
    )
    .context("failed to initialize the AdamW optimizer")?;

    let lengths: Vec<usize> = encoded
        .iter()
        .map(|scored| scored.pair.chosen.len().max(scored.pair.rejected.len()))
        .collect();
    let scale = batch::divisor(options.batch, options.accumulation);
    let steps_per_epoch =
        batch::steps_per_epoch(encoded.len(), options.batch, options.accumulation);
    let total_steps = steps_per_epoch * options.epochs;
    let mut order: Vec<usize> = encoded.iter().enumerate().map(|(slot, _)| slot).collect();
    let mut losses: Vec<f32> = Vec::with_capacity(total_steps);
    let mut epoch_summary = Summary::default();

    for (epoch, ()) in std::iter::repeat_n((), options.epochs).enumerate() {
        // Reseeded per epoch, as dpo is, so a run is reproducible from `seed`
        // alone; the same generator draws each forward's direction.
        let mut rng = StdRng::seed_from_u64(options.seed + epoch as u64);
        order.shuffle(&mut rng);
        let mut summary = Summary::default();
        for (index, plan) in batch::plan(&order, &lengths, options.batch, options.accumulation)
            .into_iter()
            .enumerate()
        {
            let step = epoch * steps_per_epoch + index;
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                step,
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
                let mut rows: Vec<&[u32]> = Vec::new();
                for &slot in forward {
                    rows.push(&encoded[slot].pair.chosen);
                    rows.push(&encoded[slot].pair.rejected);
                }
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
            losses.push(mean);
            workflow::progress(format!(
                "epoch {epoch} step {step} of {total_steps} loss {mean:.4} accuracy {:.3}",
                summary.accuracy()
            ));
        }
        epoch_summary = summary;
    }

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
        steps: losses.len(),
        first_loss: losses.first().copied(),
        final_loss: losses.last().copied(),
        mean_final_epoch_loss: epoch_summary.mean_loss(),
        accuracy: epoch_summary.accuracy(),
        mean_reward_margin: epoch_summary.mean_margin(),
        vector_norm,
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        batch: options.batch,
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
