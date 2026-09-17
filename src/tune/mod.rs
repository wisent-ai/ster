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
//!
//! The four objectives live in `objective`, the two operations that read or
//! rewrite a trained adapter live in `artifact`, and what more than one of
//! them needs is here and in `preflight`, `examples` and `batch`.

mod artifact;
mod batch;
mod examples;
mod objective;
mod preflight;

pub use artifact::{
    evaluate, merge, warn_on_provenance, EvaluateOptions, EvaluateReport, EvaluatedExample,
    MergeReport,
};
pub use examples::{Example, ExampleSet};
pub use objective::{
    decide, dpo, grpo, reward, sft, DecideOptions as DecideTuneOptions, DecideReport, DpoLoss,
    DpoOptions, DpoReport, GrpoIteration, GrpoOptions, GrpoReport, Reward, RewardHead, RewardModel,
    RewardOptions, RewardReport, SftOptions, SftReport,
};

use crate::lora;

/// The cosine schedule decays to this fraction of the peak learning rate
/// rather than to zero. LoRA runs are short; a schedule that reaches zero
/// spends a meaningful share of its last steps not learning at all.
const DECAY_FLOOR: f64 = 0.1;


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
