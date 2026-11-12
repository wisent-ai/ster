//! How what every trainer shares fails once its model is loaded: the
//! adapters it trains, the data that fits its sequence limit, the
//! fingerprint of that data, and the device taking each step
//! (src/tune/preflight.rs, checkpoint/). A tensor the device cannot
//! allocate, a score it cannot compute and a step it cannot take are the
//! cause `ster.tensor` declares for Candle's error kind, naming the tensor,
//! the score or the step: a device out of memory and a defect of Ster's own
//! tensors read apart.

use wisent_errors::{Cause, Code, Operation};

use super::super::refuse;
use super::super::runtime::encode::ENCODE;
use super::super::tensor::{self, TENSOR};
use super::DOCS;

/// The refusal of a run none of whose `unit`s fits `limit` tokens, `lengths`
/// the token count of each: `purpose` is what it would have done with them
/// (`train on`, `evaluate`, `measure`).
pub fn nothing_fits(
    unit: &str,
    purpose: &str,
    lengths: impl IntoIterator<Item = usize>,
    limit: usize,
) -> anyhow::Error {
    let (count, shortest) = lengths
        .into_iter()
        .fold((0usize, usize::MAX), |(count, shortest), length| {
            (count + 1, shortest.min(length))
        });
    refuse(
        &NOTHING_FITS,
        &[
            ("unit", unit),
            ("purpose", purpose),
            ("count", &count.to_string()),
            ("shortest", &shortest.to_string()),
            ("limit", &limit.to_string()),
        ],
    )
}

/// The refusal of `name`, of shape `dims`, the device could not allocate.
pub fn unallocatable(name: &str, dims: &[usize], error: candle_core::Error) -> anyhow::Error {
    tensor::computing(&format!("allocating {name}, a tensor of shape {dims:?}"), error)
}

/// The refusal of a forward pass that scores `what` without training, when
/// the device fails it: `error` itself when a cause already declares it.
pub fn score_failed(what: &str, error: anyhow::Error) -> anyhow::Error {
    tensor::within(error, &format!("scoring {what}"))
}

pub static DATA_UNREADABLE: Cause = Cause {
    id: "data-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "fingerprinting {path} for the run's checkpoints",
    cause: "the file the run trains on was read at the start and cannot be read again: it was moved, removed or made unreadable while the run started",
    observed: "path: {path} ({state}); reading it again: error kind {kind}; the system answered: {error}",
    remedy: "leave {path} in place and readable while the run starts, then run again",
    test: "tests/tuning/settings.sh::data_removed_while_the_run_starts_is_refused",
};

pub static NOTHING_FITS: Cause = Cause {
    id: "nothing-fits",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "encoding the {unit}s to {purpose}",
    cause: "every {unit} is longer than the sequence limit, so there is nothing to {purpose}",
    observed: "{count} {unit}s, the shortest {shortest} tokens; the limit is {limit} tokens, the lower of --max-sequence and the model's context",
    remedy: "give --max-sequence at least {shortest}, or shorten the {unit}s",
    test: "tests/tuning/settings.sh::a_limit_shorter_than_every_example_is_refused",
};

pub static SIDE_UNPREDICTABLE: Cause = Cause {
    id: "side-unpredictable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "encoding pair {index}",
    cause: "pair {index} {side} side encodes to one token, so there is nothing to predict",
    observed: "{side} side {text} encodes to the token ids {tokens}",
    remedy: "give pair {index}'s {side} side text this model's tokenizer turns into at least one token, or remove the pair",
    test: "tests/tuning/settings.sh::a_side_the_tokenizer_drops_is_refused",
};

pub static STEP_EMPTY: Cause = Cause {
    id: "step-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "taking optimizer step {step}",
    cause: "the step was planned with {forwards} forwards holding {units} {unit}s, so no loss was summed for its backward pass: the planner cuts every step from a non-empty slice of the epoch's order, so this is a defect of Ster, not of the data",
    observed: "step {step}: {forwards} forwards, {units} {unit}s, {tensors} trained tensors; nothing was written for this step",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the command and its --batch-size and --accumulation; the run's earlier checkpoints are left as they were",
    test: "tests/tuning/settings.sh::a_step_planned_without_units_is_refused",
};

/// What every trainer does once its model is loaded and before its loss is
/// its own: open the adapters, fingerprint the data, encode what fits, and
/// take each optimizer step.
pub static RUN: Operation = Operation {
    id: "ster.tune.run",
    entries: &[],
    uses: &[&ENCODE, &TENSOR],
    summary: "open the adapters a loaded model trains, fingerprint the data for its checkpoints, encode every unit that fits the sequence limit, score what the run measures without training, and take each optimizer step on the device",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &DATA_UNREADABLE,
        &NOTHING_FITS,
        &SIDE_UNPREDICTABLE,
        &STEP_EMPTY,
    ],
};
