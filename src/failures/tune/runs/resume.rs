//! How continuing a training run from its checkpoint fails
//! (src/tune/checkpoint): finding the newest checkpoint, telling it is one of
//! this run, and reading its tensors and position back. A state or tensors
//! file that cannot be read, or is not JSON or safetensors, is the
//! `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;
use super::checkpoint::LATEST;
use crate::failures::file::FILE;

/// What a checkpoint state is, for the `ster.file` refusals that name it.
pub const STATE_FORM: &str = "the state.json a ster tune run writes into each step directory of its --checkpoint";

pub static NO_CHECKPOINT: Cause = Cause {
    id: "no-checkpoint",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "finding the newest checkpoint in --resume {directory}",
    cause: "--resume {directory} holds no checkpoint: no latest pointer in it names one, so no run checkpointed there",
    observed: "--resume {directory} ({directory_state}); {pointer}: {state}",
    remedy: "give --resume the --checkpoint directory of the run to continue (`ls -ln {directory}` shows what it holds), or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::resuming_a_directory_without_a_checkpoint_is_refused",
};

pub static STATE_MISSING: Cause = Cause {
    id: "state-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading checkpoint state {path}",
    cause: "the step directory latest names holds no state.json: Ster did not finish writing it there, or it was edited by hand",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "resume from a --checkpoint directory whose latest names a step directory Ster wrote (`ls -ln {parent}` shows what this one holds), or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::a_checkpoint_without_its_state_is_refused",
};

pub static KIND_FOREIGN: Cause = Cause {
    id: "kind-foreign",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading checkpoint state {path}",
    cause: "{path} says it is {kind}, not a ster tune checkpoint",
    observed: "kind: {kind}",
    remedy: "resume from a checkpoint Ster wrote, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::a_state_of_another_kind_is_refused",
};

pub static FOREIGN_RUN: Cause = Cause {
    id: "foreign-run",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "continuing from checkpoint {checkpoint}",
    cause: "checkpoint {checkpoint} belongs to another run, so continuing it would train an adapter no single run produced",
    observed: "{differing}",
    remedy: "run with the model, data and options the checkpoint records, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::a_checkpoint_of_another_run_is_refused",
};

pub static TENSOR_ABSENT: Cause = Cause {
    id: "tensor-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the tensors of checkpoint {checkpoint}",
    cause: "{path} holds no {key}, which this run trains",
    observed: "missing: {key}",
    remedy: "resume from the checkpoint this run wrote, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_other_targets_are_refused",
};

pub static TENSOR_UNFIT: Cause = Cause {
    id: "tensor-unfit",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the tensors of checkpoint {checkpoint}",
    cause: "{key} of {path} has another shape than this run trains: the checkpoint was written with another rank, target width or model",
    observed: "{key}: stored {stored}, trained {trained}",
    remedy: "resume from the checkpoint this run wrote, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_another_rank_are_refused",
};

pub static TENSOR_DTYPE: Cause = Cause {
    id: "tensor-dtype",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the tensors of checkpoint {checkpoint}",
    cause: "{key} of {path} is stored in another dtype than this run trains it in: the checkpoint was written at another --precision",
    observed: "{key}: stored {stored}, trained {trained}",
    remedy: "resume with the --precision the checkpoint was written at, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_another_dtype_are_refused",
};

pub static TENSOR_EXTRA: Cause = Cause {
    id: "tensor-extra",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the tensors of checkpoint {checkpoint}",
    cause: "{path} holds {key}, which this run does not train",
    observed: "extra: {key}",
    remedy: "resume from the checkpoint this run wrote, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::checkpoint_tensors_of_more_targets_are_refused",
};

pub static POSITION_PAST_PASS: Cause = Cause {
    id: "position-past-pass",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "continuing the {objective} run from its checkpoint",
    cause: "the checkpoint's position stands {within} steps into pass {pass}, and this run plans {steps} steps for that pass: the state was edited, or the data, batch or accumulation it was planned with changed",
    observed: "{position}; this run's pass {pass}: {steps} steps",
    remedy: "resume with the data and settings the checkpoint was written with, from an earlier --checkpoint directory, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::a_position_past_its_pass_is_refused",
};

pub static RUNNING_FOREIGN: Cause = Cause {
    id: "running-report-foreign",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "continuing the {objective} run from its checkpoint",
    cause: "the running report the checkpoint's position carries is not the one a {objective} run keeps: the state was edited, or another objective wrote it",
    observed: "{position}; the report reader answered: {error}",
    remedy: "resume from a checkpoint a {objective} run wrote, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::a_running_report_of_another_form_is_refused",
};

pub static EPOCH_LOSSLESS: Cause = Cause {
    id: "epoch-lossless",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "closing epoch {epoch} of the resumed {objective} run",
    cause: "the checkpoint's position puts the run at the end of epoch {epoch}, and its running report records no step of that epoch, so the epoch has no mean loss: the state was edited",
    observed: "{position}; steps recorded for the epoch: 0",
    remedy: "resume from an earlier --checkpoint directory, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::an_epoch_the_report_records_no_step_of_is_refused",
};

pub static RUN_LOSSLESS: Cause = Cause {
    id: "run-lossless",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reporting the resumed {objective} run",
    cause: "the resumed run took its last step, and the checkpoint's running report records no first or final loss for the steps its position says were taken: the state was edited",
    observed: "{position}; first loss, final loss or epoch means absent from the report",
    remedy: "resume from an earlier --checkpoint directory, or start the run without --resume",
    test: "tests/tuning/runs/checkpoint.sh::a_report_without_its_losses_is_refused",
};

/// Continuing a run from the newest checkpoint `--resume` names.
pub static RESUME: Operation = Operation {
    id: "ster.tune.resume",
    entries: &[],
    uses: &[&LATEST, &FILE],
    summary: "read the newest checkpoint of --resume, refuse one another run wrote, and put its tensors, optimizer moments, learning rates and position back into this run",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &NO_CHECKPOINT,
        &STATE_MISSING,
        &KIND_FOREIGN,
        &FOREIGN_RUN,
        &TENSOR_ABSENT,
        &TENSOR_UNFIT,
        &TENSOR_DTYPE,
        &TENSOR_EXTRA,
        &POSITION_PAST_PASS,
        &RUNNING_FOREIGN,
        &EPOCH_LOSSLESS,
        &RUN_LOSSLESS,
    ],
};
