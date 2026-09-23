//! Every value a request may leave out, and the reason each one is what it
//! is. A default that is not written down here is a number nobody chose.

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::Precision;

/// Steering needs a local model, but writing pair text does not, so the
/// hosted route is opt-in and every existing client keeps the local one.
pub(super) fn default_generator() -> String {
    "local".to_owned()
}

pub(super) fn default_device() -> String {
    "cpu".to_owned()
}

pub(super) fn default_layers() -> String {
    "all".to_owned()
}

pub(super) fn default_method() -> String {
    "caa".to_owned()
}

pub(super) fn default_strength() -> f64 {
    1.0
}

pub(super) fn default_max_new_tokens() -> usize {
    128
}

pub(super) fn default_seed() -> u64 {
    42
}

pub(super) fn default_dedupe_bits() -> u32 {
    3
}

pub(super) fn default_dedupe_bands() -> u32 {
    8
}

pub(super) fn default_refusal_threshold() -> f32 {
    0.5
}

pub(super) fn default_retry_multiplier() -> usize {
    3
}

/// Synthesis answers are one or two sentences, so it keeps a tighter token
/// budget than the general generate operation.
pub(super) fn default_synthesis_max_new_tokens() -> usize {
    96
}

/// Synthesis needs sampling: at zero temperature every attempt would replay
/// the same constant prompt and the run would dedupe to a single pair.
pub(super) fn default_synthesis_temperature() -> f64 {
    0.9
}

pub(super) fn default_top_p() -> f64 {
    0.95
}

pub(super) fn default_rank() -> usize {
    8
}

/// LoRA scales every update by alpha over rank, so sixteen over the default
/// rank of eight is the two-times scale the LoRA papers train with.
pub(super) fn default_alpha() -> f64 {
    16.0
}

/// Query and value are the projections the LoRA papers adapt first, and the
/// cheapest pair that still moves behaviour.
pub(super) fn default_targets() -> String {
    "query,value".to_owned()
}

pub(super) fn default_epochs() -> usize {
    1
}

pub(super) fn default_learning_rate() -> f64 {
    1e-4
}

pub(super) fn default_accumulation() -> usize {
    8
}

/// Examples longer than this are skipped rather than truncated; a cut
/// completion would teach the model to stop early.
pub(super) fn default_max_sequence() -> usize {
    512
}

/// Apply the model's own conversation format when it publishes one. An
/// instruct checkpoint is the common case and raw text is wrong for it, so
/// the default is the setting that is right more often; `off` restores the
/// raw-text encoding a base model wants.
pub(super) fn default_chat_template() -> String {
    "auto".to_owned()
}

/// One sequence per forward: the unbatched pass every run recorded before
/// batching existed, so a client that does not ask keeps its numbers.
pub(super) fn default_batch_size() -> usize {
    1
}

/// Single precision, which is what every recorded run used. Half precision is
/// opt-in because it changes the numbers a client may be comparing against.
pub(super) fn default_precision() -> String {
    "f32".to_owned()
}

/// Records the dtype the base weights were mapped at in a run's own report,
/// beside the chat-template decision and for the same reason: two runs of the
/// same request at different precisions produce different losses, and a report
/// that does not say which one made it is not comparable with the other.
pub(in crate::request) fn note_precision(report: &mut Value, precision: Precision) -> Result<()> {
    report
        .as_object_mut()
        .context("a run report must be a JSON object to record its precision")?
        .insert("precision".to_owned(), json!(precision.name()));
    Ok(())
}

/// The strength of the pull back toward the frozen reference, at the value the
/// DPO paper reports across its settings.
pub(super) fn default_beta() -> f64 {
    0.1
}

/// The sigmoid objective the DPO paper derives; `ipo` is the squared-error
/// alternative over length-normalized log-probabilities.
pub(super) fn default_preference_loss() -> String {
    "dpo".to_owned()
}

/// The offline reward: a completion's sampled-token count, which needs no
/// judge and no artifact, so the loop is runnable the first time it is asked
/// for.
pub(super) fn default_reward() -> String {
    "length".to_owned()
}

/// Four completions per prompt: enough for a baseline that is not the sample
/// itself, cheap enough that a first run finishes.
pub(super) fn default_group() -> usize {
    4
}

pub(super) fn default_iterations() -> usize {
    1
}

/// The KL weight the GRPO paper reports; smaller than a preference loss's beta
/// because it is a penalty per token rather than a scale on the whole margin.
pub(super) fn default_kl_beta() -> f64 {
    0.04
}

/// One prompt group is already `group` sequences, so a step per group is the
/// natural unit and this default is one where the other trainers use eight.
pub(super) fn default_group_accumulation() -> usize {
    1
}

pub(super) fn default_grpo_max_new_tokens() -> usize {
    64
}

/// Policy optimization needs sampling: at zero temperature every draw in a
/// group would be the same completion and the baseline would have nothing to
/// compare against.
pub(super) fn default_grpo_temperature() -> f64 {
    0.9
}

