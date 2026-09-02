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
use serde::Serialize;

use super::{
    EncodedPair, Preflight, Trainable, batch, encode_pairs, pair_set_label, schedule,
    sequence_logprob, softplus,
};
use crate::{artifact::PairSet, lora, model::Route, runtime::Runtime, workflow};

/// Which preference objective the log-ratio margin is fed into.
///
/// Both read the same pair set, take the same forward passes, and differ only
/// in the scalar function applied at the very end, which is why supporting the
/// second costs one match arm rather than a second trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpoLoss {
    Dpo,
    Ipo,
}

impl DpoLoss {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dpo" => Ok(Self::Dpo),
            "ipo" => Ok(Self::Ipo),
            _ => bail!("unknown preference loss {value:?}; expected dpo or ipo"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Dpo => "dpo",
            Self::Ipo => "ipo",
        }
    }

    /// Whether the log-probabilities are divided by the number of tokens they
    /// sum over.
    ///
    /// DPO's derivation is over whole-sequence log-probabilities and length
    /// enters the objective on purpose. IPO replaces the sigmoid with a squared
    /// error against a fixed target, and a target that a long sequence reaches
    /// by length alone is not a preference signal, so IPO scores the mean
    /// per-token log-probability instead.
    fn length_normalized(self) -> bool {
        matches!(self, Self::Ipo)
    }
}

#[derive(Debug, Clone)]
pub struct DpoOptions {
    pub spec: lora::Spec,
    pub loss: DpoLoss,
    /// How hard the reference pulls back. Small beta lets the policy move far
    /// from the reference before the loss objects; the 0.1 default is the value
    /// the DPO paper reports across its settings.
    pub beta: f64,
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
pub struct DpoReport {
    pub loss: String,
    pub beta: f64,
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
    /// The share of pairs the policy already prefers the chosen side on,
    /// measured over the final epoch — so it describes the adapter that was
    /// written, not an average over a policy that was still moving.
    pub accuracy: f32,
    pub mean_reward_margin: f32,
    pub mean_chosen_reward: f32,
    pub mean_rejected_reward: f32,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub layers: Vec<usize>,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
}

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
    let Trainable { spec, vars, tensors, parameters, limit } = Preflight {
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
        ParamsAdamW { lr: options.learning_rate, ..Default::default() },
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
                let logits = runtime.forward_train_rows(&rows)?;
                for (position, &slot) in forward.iter().enumerate() {
                    let scored = &encoded[slot];
                    let chosen = batch::row(&logits, position * 2, scored.pair.chosen.len())?;
                    let rejected =
                        batch::row(&logits, position * 2 + 1, scored.pair.rejected.len())?;
                    let value = step_loss(runtime, scored, &chosen, &rejected, options)
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
        targets: spec.targets.iter().map(|target| target.name().to_owned()).collect(),
        layers: spec.layers.clone(),
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        batch: options.batch,
    })
}

/// One tokenized pair with the frozen reference's opinion of both sides.
struct Scored {
    pair: EncodedPair,
    chosen_reference: f64,
    rejected_reference: f64,
}

impl Scored {
    /// The reference values are filled in by [`reference_scores`] before the
    /// optimizer exists; zero is not a plausible log-probability and would
    /// surface immediately as a first loss that is not `ln 2`.
    fn new(pair: EncodedPair) -> Self {
        Self { pair, chosen_reference: 0.0, rejected_reference: 0.0 }
    }
}

/// Fills in each pair's frozen-reference log-probabilities.
///
/// Run once, before the optimizer exists. The reference never changes, so
/// re-deriving these every epoch would be `2 * pairs * (epochs - 1)` forward
/// passes spent recomputing constants.
fn reference_scores(runtime: &Runtime, encoded: &mut [Scored], pairs: usize) -> Result<()> {
    let total = encoded.len();
    workflow::progress(format!("scoring {total} pairs under the frozen reference model"));
    let device = runtime.device();
    let mut scored = 0usize;
    // Batched on the same knob the training loop uses, in input order: the
    // reference is a constant and nothing here is shuffled, so the only thing
    // a group decides is how many rows share a kernel launch.
    for group in encoded.chunks_mut(pairs.max(1)) {
        let mut rows: Vec<&[u32]> = Vec::with_capacity(group.len() * 2);
        for pair in group.iter() {
            rows.push(&pair.pair.chosen);
            rows.push(&pair.pair.rejected);
        }
        let logits = runtime.forward_scored_rows(&rows, Route::Base)?;
        for (position, pair) in group.iter_mut().enumerate() {
            let row = batch::row(&logits, position * 2, pair.pair.chosen.len())?;
            pair.chosen_reference =
                sequence_logprob(&row, &pair.pair.chosen, 1, device)?.to_scalar::<f32>()? as f64;
            let row = batch::row(&logits, position * 2 + 1, pair.pair.rejected.len())?;
            pair.rejected_reference =
                sequence_logprob(&row, &pair.pair.rejected, 1, device)?.to_scalar::<f32>()?
                    as f64;
            scored += 1;
            workflow::progress(format!("reference pair {scored}/{total}"));
        }
    }
    Ok(())
}

/// The loss for one pair, plus the scalars the report is built from.
struct Step {
    tensor: Tensor,
    loss: f64,
    chosen_reward: f64,
    rejected_reward: f64,
}

/// One pair's contribution: two scored rows, one margin, one loss.
///
/// The logits arrive already read out of whatever forward produced them, so
/// this function is identical whether one pair went through the model or
/// eight did.
fn step_loss(
    runtime: &Runtime,
    scored: &Scored,
    chosen_logits: &Tensor,
    rejected_logits: &Tensor,
    options: &DpoOptions,
) -> Result<Step> {
    let chosen = policy_log_ratio(
        runtime,
        chosen_logits,
        &scored.pair.chosen,
        scored.chosen_reference,
        options,
    )?;
    let rejected = policy_log_ratio(
        runtime,
        rejected_logits,
        &scored.pair.rejected,
        scored.rejected_reference,
        options,
    )?;
    let margin = (&chosen - &rejected)?;
    let tensor = match options.loss {
        // -log sigmoid(beta * margin), written as the softplus that does not
        // overflow when the policy is already confident.
        DpoLoss::Dpo => softplus(&(&margin * -options.beta)?)?,
        // Equation 17 of the IPO paper: a squared error against the fixed
        // target 1 / (2 * beta) rather than a sigmoid, which is what stops the
        // objective from being satisfied by driving the margin to infinity.
        DpoLoss::Ipo => (&margin - 1.0 / (2.0 * options.beta))?.sqr()?,
    };
    // The implicit reward DPO derives is beta times the log-ratio. Reading it
    // off the same two tensors the loss was built from — rather than
    // recomputing it — is what makes the reported accuracy the accuracy of the
    // step that was actually taken.
    Ok(Step {
        loss: tensor.to_scalar::<f32>()? as f64,
        chosen_reward: options.beta * chosen.to_scalar::<f32>()? as f64,
        rejected_reward: options.beta * rejected.to_scalar::<f32>()? as f64,
        tensor,
    })
}

/// The policy's log-probability of `ids` minus the reference's, normalized for
/// the objective that will consume it.
fn policy_log_ratio(
    runtime: &Runtime,
    logits: &Tensor,
    ids: &[u32],
    reference: f64,
    options: &DpoOptions,
) -> Result<Tensor> {
    let policy = sequence_logprob(logits, ids, 1, runtime.device())?;
    let ratio = (policy - reference)?;
    if options.loss.length_normalized() {
        // `sequence_logprob` scores every token after the begin-of-sequence
        // marker, so that is the count the mean divides by.
        return Ok((ratio / (ids.len() - 1) as f64)?);
    }
    Ok(ratio)
}

/// Running totals over one epoch.
#[derive(Debug, Default)]
struct Summary {
    pairs: usize,
    loss: f64,
    correct: usize,
    chosen_reward: f64,
    rejected_reward: f64,
}

impl Summary {
    fn record(&mut self, step: &Step) {
        self.pairs += 1;
        self.loss += step.loss;
        self.chosen_reward += step.chosen_reward;
        self.rejected_reward += step.rejected_reward;
        if step.chosen_reward > step.rejected_reward {
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
        self.mean(self.chosen_reward)
    }

    fn mean_rejected(&self) -> f32 {
        self.mean(self.rejected_reward)
    }

    fn mean_margin(&self) -> f32 {
        self.mean(self.chosen_reward - self.rejected_reward)
    }

    fn accuracy(&self) -> f32 {
        self.correct as f32 / self.pairs.max(1) as f32
    }
}
