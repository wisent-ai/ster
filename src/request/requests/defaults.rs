//! Every value a request may leave out, and the reason each one is what it
//! is. Only choices with a neutral or documented meaning live here; every
//! training, sampling and filtering number is the caller's to state, and a
//! request without one is refused by name.

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::Precision;

pub(super) fn default_device() -> String {
    "cpu".to_owned()
}

pub(super) fn default_layers() -> String {
    "all".to_owned()
}

pub(super) fn default_method() -> String {
    "caa".to_owned()
}

/// Query and value are the projections the LoRA papers adapt first, and the
/// cheapest pair that still moves behaviour.
pub(super) fn default_targets() -> String {
    "query,value".to_owned()
}

/// Apply the model's own conversation format when it publishes one. An
/// instruct checkpoint is the common case and raw text is wrong for it, so
/// the default is the setting that is right more often; `off` restores the
/// raw-text encoding a base model wants.
pub(super) fn default_chat_template() -> String {
    "auto".to_owned()
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

/// One assistant turn: a single completion per prompt, which is what a body
/// without the field has always asked for.
pub(super) fn default_turns() -> usize {
    1
}
