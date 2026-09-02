//! tune.rs — training the only weights Ster owns.
//!
//! Everything else in Ster is forward-only: it reads hidden states, fits a
//! direction over them, and adds that direction back during decode. Nothing
//! there needs a gradient with respect to a weight. This module is the one
//! place that does.
//!
//! It is a root, not an objective. Each objective is a sibling file that owns
//! only its own loss, options and report — [`sft`] a masked next-token
//! cross-entropy, [`dpo`] the preference losses, [`reward`] a Bradley-Terry
//! head, [`grpo`] a group-relative policy gradient — and everything more than
//! one of them needs lives here: the preflight checks, the input formats, the
//! tokenizers, the log-probability readers, the softplus and the learning-rate
//! schedule. None of the four is privileged; they are re-exported flat, so a
//! caller writes `tune::sft` and `tune::grpo` side by side and the file split
//! stays an implementation detail.
//!
//! Two properties hold across all four and are structural rather than
//! conventional:
//!
//! * **Only what the run created trains.** `Runtime::load_trainable` maps the
//!   base weights read-only and registers nothing but the low-rank pairs in the
//!   `VarMap`; a reward run adds its scalar head to that same map. The
//!   optimizer is constructed from `varmap.all_vars()`, so there is no base
//!   weight it could reach even if a loss asked for one.
//! * **The frozen reference is free.** Because `B` starts at zero, the base
//!   weights *are* the model every adapter started as. Any objective that needs
//!   a reference — the preference losses, the policy gradient's KL — gets it by
//!   skipping the adapters for one pass rather than by loading a second
//!   checkpoint.

mod batch;
mod dpo;
mod evaluate;
mod grpo;
mod merge;
mod reward;
mod sft;

pub use dpo::{DpoLoss, DpoOptions, DpoReport, dpo};
pub use evaluate::{EvaluateOptions, EvaluateReport, EvaluatedExample, evaluate};
pub use grpo::{GrpoIteration, GrpoOptions, GrpoReport, Reward, grpo};
pub use merge::{MergeReport, merge};
pub use reward::{RewardHead, RewardModel, RewardOptions, RewardReport, reward};
pub use sft::{SftOptions, SftReport, sft};

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor, Var};
use candle_nn::VarMap;
use serde::Deserialize;

use crate::{artifact::PairSet, lora, runtime::Runtime, workflow};

/// The cosine schedule decays to this fraction of the peak learning rate
/// rather than to zero. LoRA runs are short; a schedule that reaches zero
/// spends a meaningful share of its last steps not learning at all.
const DECAY_FLOOR: f64 = 0.1;

// MARK: - Shared trainer setup

/// Everything an adapter trainer checks before it touches a model, and the
/// four words it says it in.
///
/// Four objectives run the same five checks against the same numbers. Copying
/// them would let one copy gain a check the others silently lacked, so they
/// live here once, and the vocabulary is what makes one set of checks refuse
/// in four voices:
///
/// * `subject` opens every sentence.
/// * `unit` is one item of the training set — an example, a pair, a prompt.
/// * `pass` is one traversal of it. Three objectives call that an epoch;
///   policy optimization calls it an iteration, because it re-samples rather
///   than re-reading and is not seeing the same data twice.
/// * `noun` names the tensors in the opening progress line. Most runs train
///   adapters and nothing else, but a reward run also trains a scalar head,
///   and calling that an adapter would be wrong.
///
/// The supervised sentences are the ones the runbook quotes and this
/// vocabulary keeps them byte-identical.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Preflight<'a> {
    pub subject: &'a str,
    pub unit: &'a str,
    pub pass: &'a str,
    pub noun: &'a str,
    pub epochs: usize,
    pub accumulation: usize,
    /// Sequences folded into one forward pass. One is the unbatched pass every
    /// objective took before batching existed, which is why it is the default
    /// rather than a tuned number: a batch changes only how many sequences
    /// share a kernel launch, never what the loss is.
    pub batch: usize,
    pub learning_rate: f64,
    pub max_sequence: usize,
}

/// What a passed preflight hands the trainer.
pub(crate) struct Trainable {
    /// The spec with `layers` pinned to this model's real indices.
    pub spec: lora::Spec,
    pub vars: Vec<Var>,
    pub tensors: usize,
    pub parameters: usize,
    /// The longest sequence this run will score: the operator's limit, clamped
    /// to what the rotary tables actually cover.
    pub limit: usize,
}

impl Preflight<'_> {
    pub(crate) fn open(
        &self,
        runtime: &Runtime,
        varmap: &VarMap,
        spec: &lora::Spec,
    ) -> Result<Trainable> {
        if self.epochs == 0 {
            bail!("{} requires at least one {}", self.subject, self.pass);
        }
        if self.accumulation == 0 {
            bail!(
                "{} requires an accumulation of at least one {}",
                self.subject,
                self.unit
            );
        }
        if self.batch == 0 {
            bail!("{} requires a batch of at least one {}", self.subject, self.unit);
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            bail!("{} requires a finite learning rate above zero", self.subject);
        }
        if self.max_sequence < 2 {
            bail!("max_sequence must be at least two tokens, so that one token can predict another");
        }
        let layer_count = runtime.layer_count();
        spec.validate(layer_count)?;
        let spec = spec.resolved(layer_count);

        // Nothing in the map is a base weight: the base was mapped read-only
        // and never registered, so the map holds the adapters and, on a reward
        // run, the scalar head. Handing the whole map to AdamW is therefore the
        // same statement as "train only what this run created".
        let vars = varmap.all_vars();
        if vars.is_empty() {
            bail!("this run has no trainable adapters; load the model with load_trainable");
        }
        let tensors = vars.len();
        let parameters: usize = vars.iter().map(|var| var.elem_count()).sum();
        workflow::progress(format!(
            "training {tensors} {} ({parameters} parameters) at rank {} across {} layers",
            self.noun,
            spec.rank,
            spec.layers.len()
        ));
        let limit = self.max_sequence.min(runtime.context_length());
        Ok(Trainable { spec, vars, tensors, parameters, limit })
    }
}

/// The log-probability the model assigns to each of `ids[start..]`, as a
/// `[len(ids) - start]` vector.
///
/// `logits` is `[1, n, vocab]` and the distribution that predicts token `t`
/// sits at index `t - 1`, so the scored window is the logit rows
/// `[start - 1, n - 1)` against the targets `ids[start..]`. `start` is
/// therefore at least one, and it always is: every tokenizer path in Ster
/// prepends a begin-of-sequence marker, so position zero is never a token
/// anyone asked the model to predict.
///
/// Per token rather than summed, because the objectives disagree about what to
/// do with them: a preference loss wants the sum, a policy-gradient loss wants
/// a per-token ratio and a per-token divergence. Summing is the caller's one
/// extra line; unsumming is impossible.
pub(crate) fn token_logprobs(
    logits: &Tensor,
    ids: &[u32],
    start: usize,
    device: &Device,
) -> Result<Tensor> {
    if start == 0 {
        bail!("a scored sequence must begin with at least one context token");
    }
    let scored = ids.len().saturating_sub(start);
    if scored == 0 {
        bail!("a scored sequence must end with at least one predicted token");
    }
    let (_, positions, vocab) = logits.dims3()?;
    if positions != ids.len() {
        bail!(
            "the forward pass returned {positions} positions for {} tokens",
            ids.len()
        );
    }
    let window = logits.narrow(1, start - 1, scored)?.reshape((scored, vocab))?;
    let log_probabilities = candle_nn::ops::log_softmax(&window, candle_core::D::Minus1)?;
    let targets = Tensor::new(&ids[start..], device)?.reshape((scored, 1))?;
    Ok(log_probabilities.gather(&targets, 1)?.squeeze(1)?)
}

/// The log-probability of the whole window, as one scalar.
pub(crate) fn sequence_logprob(
    logits: &Tensor,
    ids: &[u32],
    start: usize,
    device: &Device,
) -> Result<Tensor> {
    Ok(token_logprobs(logits, ids, start, device)?.sum_all()?)
}

/// `log(1 + exp(x))`, computed the way that does not overflow.
///
/// Every preference objective in Ster ends in `-log sigmoid(z)`, which is
/// `softplus(-z)`. The direct form is infinite in F32 from about x = 89, while
/// the true value there is 89. Pulling the positive part out through `relu`
/// leaves `log(1 + exp(-|x|))`, whose argument is always in (1, 2], and every
/// op in the expression records a backward node.
pub(crate) fn softplus(x: &Tensor) -> Result<Tensor> {
    let tail = ((x.abs()?.neg()?.exp()? + 1.0)?).log()?;
    Ok((x.relu()? + tail)?)
}

/// One preference pair, tokenized.
pub(crate) struct EncodedPair {
    /// The pair's index in the original set, so a refusal names the entry the
    /// operator wrote rather than a position in a filtered list.
    pub index: usize,
    pub chosen: Vec<u32>,
    pub rejected: Vec<u32>,
}

/// Tokenizes both sides of every pair, dropping the ones that do not fit.
///
/// Over-long pairs are skipped rather than truncated for the reason supervised
/// fine-tuning skips over-long examples: a cut sequence is a different
/// sequence, and preferring a truncation of the chosen side over a truncation
/// of the rejected side is not the preference the operator stated. Both sides
/// go or neither does — half a pair states no preference at all.
///
/// Each side goes through `Runtime::encode_response` rather than the raw
/// encoder, so a run that asked for the model's chat template compares two
/// assistant turns wearing the markers inference will put around them, and a
/// run that did not gets byte-identical ids to every release before this one.
///
/// Tokenizing up front rather than per epoch is what makes the skip report
/// complete before the first gradient is taken.
pub(crate) fn encode_pairs(
    runtime: &Runtime,
    pairs: &PairSet,
    limit: usize,
) -> Result<Vec<EncodedPair>> {
    let mut encoded = Vec::with_capacity(pairs.pairs.len());
    for (index, pair) in pairs.pairs.iter().enumerate() {
        let chosen = runtime
            .encode_response(&pair.positive)
            .with_context(|| format!("pair {index} chosen side could not be encoded"))?;
        let rejected = runtime
            .encode_response(&pair.negative)
            .with_context(|| format!("pair {index} rejected side could not be encoded"))?;
        for (side, ids) in [("chosen", &chosen), ("rejected", &rejected)] {
            if ids.len() < 2 {
                bail!(
                    "pair {index} {side} side encodes to one token, so there is nothing to predict"
                );
            }
        }
        let longest = chosen.len().max(rejected.len());
        if longest > limit {
            workflow::progress(format!(
                "skipping pair {index}: {longest} tokens exceed the {limit} token limit"
            ));
            continue;
        }
        encoded.push(EncodedPair { index, chosen, rejected });
    }
    if encoded.is_empty() {
        bail!("every pair is longer than the sequence limit, so there is nothing to train on");
    }
    Ok(encoded)
}

/// How a pair set names itself in a refusal when it carries no trait name.
pub(crate) fn pair_set_label(pairs: &PairSet) -> String {
    match pairs.trait_name.trim() {
        "" => "(unnamed)".to_owned(),
        name => name.to_owned(),
    }
}

// MARK: - Example sets

#[derive(Debug, Clone, Deserialize)]
pub struct Example {
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExampleSet {
    #[serde(default)]
    pub name: String,
    pub examples: Vec<Example>,
}

impl ExampleSet {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read example set {}", path.display()))?;
        let value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid example set JSON in {}", path.display()))?;
        value.validate(&path.display().to_string())?;
        Ok(value)
    }

    /// Checks the two content invariants every consumer of an example set
    /// relies on.
    ///
    /// `label` is the identity quoted in the refusal sentence, exactly as
    /// `PairSet::validate` uses it: the loader passes the path, and a caller
    /// validating a set assembled from an API request passes whatever names it
    /// to the operator. Both sentences are published in the runbook, so they
    /// must stay byte-identical regardless of which caller triggers them.
    pub fn validate(&self, label: &str) -> Result<()> {
        if self.examples.is_empty() {
            bail!("example set {label} contains no examples");
        }
        if self
            .examples
            .iter()
            .any(|example| example.prompt.trim().is_empty() || example.completion.trim().is_empty())
        {
            bail!("example set {label} contains an empty prompt or completion");
        }
        Ok(())
    }

    /// How the set names itself in a refusal when it never came from a file.
    pub(crate) fn label(&self) -> String {
        match self.name.trim() {
            "" => "(unnamed)".to_owned(),
            name => name.to_owned(),
        }
    }
}


/// Linear warmup for `warmup` steps, then cosine decay from `base` down to
/// `DECAY_FLOOR * base` over whatever steps remain.
///
/// Warmup counts from one so that the very first step is not taken at a zero
/// learning rate, which would waste the one step whose gradient is largest.
fn schedule(base: f64, step: usize, total: usize, warmup: usize) -> f64 {
    if step < warmup {
        return base * (step + 1) as f64 / warmup as f64;
    }
    let floor = base * DECAY_FLOOR;
    let decaying = total.saturating_sub(warmup);
    if decaying <= 1 {
        return base;
    }
    let progress = ((step - warmup) as f64 / (decaying - 1) as f64).clamp(0.0, 1.0);
    floor + (base - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

// MARK: - Inspection

/// The adapter equivalent of `workflow::artifact_summary`: everything the
/// artifact knows about itself, including the shape of every tensor it
/// carries, without loading a model. Auditing which layers and projections an
/// adapter touches should not cost a multi-gigabyte mmap.
pub fn inspect(artifact: &lora::Artifact) -> serde_json::Value {
    // `Spec::scale` is the same ratio, but an artifact is not a spec and a
    // rank of zero would never have been written; guarding here keeps the
    // inspector total over files it did not produce.
    let scale = if artifact.rank == 0 {
        0.0
    } else {
        artifact.alpha / artifact.rank as f64
    };
    serde_json::json!({
        "adapter": {
            "schema_version": artifact.schema_version,
            "product": artifact.product,
            "kind": artifact.kind.name(),
            "model": artifact.model,
            "model_revision": artifact.model_revision,
            "rank": artifact.rank,
            "alpha": artifact.alpha,
            "scale": scale,
            "targets": artifact.targets.iter().map(|target| target.name()).collect::<Vec<_>>(),
            "layers": artifact.layers,
            "hidden_size": artifact.hidden_size,
            "tensors": artifact.tensors.iter().map(|(name, tensor)| serde_json::json!({
                "name": name,
                "shape": tensor.dims(),
            })).collect::<Vec<_>>(),
            "train": artifact.train,
        }
    })
}
