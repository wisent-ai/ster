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
//!   exactly `ln 2`, the same identity check the preference losses give.
//!
//! The head trains together with the adapters beneath it, in one `VarMap` and
//! under one optimizer, and is written into the same safetensors file. A head
//! is a function of the residual stream the adapters shaped; pairing one with
//! adapters it never saw would produce scores that mean nothing, so the
//! artifact does not offer that as a possibility.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarMap};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use super::{Preflight, Trainable, batch, encode_pairs, pair_set_label, schedule, softplus};
use crate::{
    artifact::PairSet,
    lora,
    runtime::{DeviceChoice, Runtime},
    workflow,
};

/// The scalar head a reward model scores with: one row, `hidden_size` wide.
#[derive(Debug, Clone)]
pub struct RewardHead {
    weight: Tensor,
}

impl RewardHead {
    /// A fresh head registered in `varmap`, so one optimizer steps it and the
    /// adapters together.
    ///
    /// Zeroed rather than drawn. A single output row has no symmetry for a
    /// random draw to break, and the Bradley-Terry gradient at zero is
    /// `-(h_chosen - h_rejected) / 2`, which is as far from zero as the two
    /// residual states are from each other — so the head learns immediately
    /// and the run needs no seed of its own.
    pub fn fresh(varmap: &VarMap, hidden: usize, device: &Device, dtype: DType) -> Result<Self> {
        let weight = varmap
            .get((1, hidden), lora::REWARD_HEAD_TENSOR, Init::Const(0.0), dtype, device)
            .with_context(|| format!("failed to create {}", lora::REWARD_HEAD_TENSOR))?;
        Ok(Self { weight })
    }

    /// The head read back out of an artifact, frozen.
    pub fn from_tensor(weight: Tensor) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!(
                "reward head has shape {dims:?}, expected one row of hidden-size weights"
            );
        }
        Ok(Self { weight })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// The score of one sequence, given `[1, sequence, hidden]`.
    ///
    /// The last position is the one that has attended to the whole sequence,
    /// so it is the only position that can score it. A batched pass hands its
    /// rows over one at a time through `batch::row`, already sliced back to
    /// each row's own length, so the last position here is always a real token
    /// and never padding.
    pub fn score(&self, hidden: &Tensor) -> Result<Tensor> {
        let (_, sequence, width) = hidden.dims3()?;
        let expected = self.weight.dim(1)?;
        if width != expected {
            bail!("reward head is {expected} wide, the model's residual stream is {width}");
        }
        let last = hidden.i((0, sequence - 1, ..))?;
        // An elementwise product folded to a scalar rather than a matmul: the
        // result is one number, and a `[1, hidden] x [hidden, 1]` matmul would
        // reshape twice to say the same thing.
        Ok((last * self.weight.squeeze(0)?)?.sum_all()?)
    }
}

/// A trained reward model, loaded and frozen: the base weights with the
/// artifact's adapters attached, and the head that reads them.
///
/// This is a second model in memory beside whatever policy is being trained,
/// and that is not an oversight to optimize away later: a judge is genuinely a
/// different model from the thing it judges. What it is not is a second copy
/// of anything — the adapters are the artifact's own, and the base weights are
/// mapped read-only exactly as every other Ster load maps them.
pub struct RewardModel {
    runtime: Runtime,
    head: RewardHead,
}

impl RewardModel {
    /// Loads the reward artifact at `path` against `model`.
    ///
    /// The artifact must declare kind `reward`; a generation adapter has no
    /// head, and attaching one here would silently score every sequence with
    /// whatever the caller passed instead.
    pub fn load(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        path: &Path,
    ) -> Result<Self> {
        let (runtime, artifact) =
            Runtime::load_artifact(model, revision, device, path, lora::Kind::Reward)?;
        let weight = artifact
            .tensors
            .get(lora::REWARD_HEAD_TENSOR)
            .with_context(|| format!("reward artifact is missing {}", lora::REWARD_HEAD_TENSOR))?
            .to_device(runtime.device())?
            .to_dtype(runtime.dtype())?;
        Ok(Self { runtime, head: RewardHead::from_tensor(weight)? })
    }

    /// The reward this model assigns to one tokenized sequence.
    ///
    /// The whole sequence goes in, prompt included: the head reads the last
    /// position, which has attended to everything before it, so a response is
    /// scored in the context it was a response to.
    pub fn score(&self, ids: &[u32]) -> Result<f64> {
        let hidden = self.runtime.forward_hidden_scored(ids)?;
        Ok(self.head.score(&hidden)?.to_scalar::<f32>()? as f64)
    }
}

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
    let Trainable { spec, vars, tensors, parameters, limit } = Preflight {
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
        ParamsAdamW { lr: options.learning_rate, ..Default::default() },
    )
    .context("failed to initialize the AdamW optimizer")?;

    // Both sides of a pair go through the same forward, so the width its rows
    // are padded to is the longer of the two.
    let lengths: Vec<usize> =
        encoded.iter().map(|pair| pair.chosen.len().max(pair.rejected.len())).collect();
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
            "epoch {}/{} mean loss {:.4} accuracy {:.3} score margin {:.4}",
            epoch + 1,
            options.epochs,
            summary.mean_loss(),
            summary.accuracy(),
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
    })
}

/// The loss for one pair, plus the scalars the report is built from.
struct Step {
    tensor: Tensor,
    loss: f64,
    chosen: f64,
    rejected: f64,
}

/// One pair's contribution: two scored rows and one Bradley-Terry loss.
///
/// The residual streams arrive already read out of whatever forward produced
/// them, so this function is identical whether one pair went through the model
/// or eight did.
fn step_loss(head: &RewardHead, chosen: &Tensor, rejected: &Tensor) -> Result<Step> {
    let chosen = head.score(chosen)?;
    let rejected = head.score(rejected)?;
    // -log sigmoid(chosen - rejected), through the softplus that survives a
    // head confident enough to overflow the direct form.
    let tensor = softplus(&(&rejected - &chosen)?)?;
    Ok(Step {
        loss: tensor.to_scalar::<f32>()? as f64,
        chosen: chosen.to_scalar::<f32>()? as f64,
        rejected: rejected.to_scalar::<f32>()? as f64,
        tensor,
    })
}

/// Running totals over one epoch.
#[derive(Debug, Default)]
struct Summary {
    pairs: usize,
    loss: f64,
    correct: usize,
    chosen: f64,
    rejected: f64,
}

impl Summary {
    fn record(&mut self, step: &Step) {
        self.pairs += 1;
        self.loss += step.loss;
        self.chosen += step.chosen;
        self.rejected += step.rejected;
        if step.chosen > step.rejected {
            self.correct += 1;
        }
    }

    /// Every mean below divides by the pair count, guarded at one so an epoch
    /// that recorded nothing reports zero rather than a JSON `NaN` no client
    /// can parse.
    fn mean(&self, total: f64) -> f32 {
        (total / self.pairs.max(1) as f64) as f32
    }

    fn mean_loss(&self) -> f32 {
        self.mean(self.loss)
    }

    fn mean_chosen(&self) -> f32 {
        self.mean(self.chosen)
    }

    fn mean_rejected(&self) -> f32 {
        self.mean(self.rejected)
    }

    fn mean_margin(&self) -> f32 {
        self.mean(self.chosen - self.rejected)
    }

    fn accuracy(&self) -> f32 {
        self.correct as f32 / self.pairs.max(1) as f32
    }
}
