//! How writing a training run's checkpoint fails (src/tune/checkpoint), and
//! the `latest` pointer writing and resuming both read. Every file and
//! directory it reads, writes, renames or removes failing is the `ster.file`
//! layer's cause, named with what the step left in place.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;
use crate::failures::file::FILE;

pub static LATEST_INVALID: Cause = Cause {
    id: "latest-invalid",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {pointer}, which names the newest checkpoint",
    cause: "it names {name}, which is not one step directory inside {directory}",
    observed: "latest: {name}",
    remedy: "write the name of one step-* directory of {directory} into {pointer}, or start the run with an empty --checkpoint directory",
    test: "tests/tuning/runs/checkpoint.sh::a_latest_pointer_naming_another_place_is_refused",
};

/// The `latest` pointer of a checkpoint directory, read before a checkpoint
/// replaces it and when a run resumes.
pub static LATEST: Operation = Operation {
    id: "ster.tune.latest",
    entries: &[],
    uses: &[&FILE],
    summary: "read the latest file of a checkpoint directory and resolve the one step directory it names",
    docs: DOCS,
    warnings: &[],
    causes: &[&LATEST_INVALID],
};

pub static CLOCK_UNREADABLE: Cause = Cause {
    id: "clock-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "naming the checkpoint written at step {step}",
    cause: "the system clock stands before 1970, so the attempt cannot be named by the time it was written",
    observed: "step: {step}; the clock answered: {error}",
    remedy: "set the system clock to the current time and run again",
    test: "tests/tuning/runs/checkpoint.sh::a_clock_before_1970_is_refused",
};

/// Writing a run's checkpoint at a step the operator named.
pub static SAVE: Operation = Operation {
    id: "ster.tune.save",
    entries: &[],
    uses: &[&LATEST, &FILE],
    summary: "write a run's trained tensors, optimizer moments and position under a new step directory of --checkpoint, name it in latest, and remove the checkpoint it replaces",
    docs: DOCS,
    warnings: &[],
    causes: &[&CLOCK_UNREADABLE],
};
