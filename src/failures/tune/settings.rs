//! How the numbers every trainer takes fail: the passes, accumulation,
//! batch, learning rate and sequence limit (src/tune/preflight.rs), refused
//! before any model, data set or checkpoint is read.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

pub static PASSES_NONE: Cause = Cause {
    id: "passes-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting {subject}",
    cause: "{subject} requires at least one {pass}",
    observed: "--{pass}s 0",
    remedy: "give --{pass}s a count of at least 1",
    test: "tests/tuning/settings.sh::no_pass_is_refused_before_the_model_loads",
};

pub static ACCUMULATION_NONE: Cause = Cause {
    id: "accumulation-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting {subject}",
    cause: "{subject} requires an accumulation of at least one {unit}",
    observed: "--accumulation {value}",
    remedy: "give --accumulation a count of at least 1; 1 steps the optimizer after every forward pass",
    test: "tests/tuning/settings.sh::no_accumulation_is_refused_before_the_model_loads",
};

pub static BATCH_NONE: Cause = Cause {
    id: "batch-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting {subject}",
    cause: "{subject} requires a batch of at least one {unit}",
    observed: "--batch-size {value}",
    remedy: "give --batch-size a count of at least 1; 1 reproduces an unbatched run byte for byte",
    test: "tests/tuning/settings.sh::no_batch_is_refused_before_the_model_loads",
};

pub static LEARNING_RATE_UNUSABLE: Cause = Cause {
    id: "learning-rate-unusable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting {subject}",
    cause: "{subject} requires a finite learning rate above zero",
    observed: "--learning-rate {value}",
    remedy: "give --learning-rate a finite number above zero, such as 0.0001",
    test: "tests/tuning/settings.sh::an_unusable_learning_rate_is_refused_before_the_model_loads",
};

pub static SEQUENCE_SHORT: Cause = Cause {
    id: "sequence-short",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting {subject}",
    cause: "max_sequence must be at least two tokens, so that one token can predict another",
    observed: "--max-sequence {value}",
    remedy: "give --max-sequence at least 2, as many tokens as the longest {unit} the run should train on",
    test: "tests/tuning/settings.sh::a_sequence_limit_below_two_is_refused_before_the_model_loads",
};

/// The numbers every trainer refuses before it reads anything.
pub static SETTINGS: Operation = Operation {
    id: "ster.tune.settings",
    entries: &[],
    uses: &[],
    summary: "check the passes, accumulation, batch, learning rate and sequence limit a training run is given, before any model, data or checkpoint is read",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &PASSES_NONE,
        &ACCUMULATION_NONE,
        &BATCH_NONE,
        &LEARNING_RATE_UNUSABLE,
        &SEQUENCE_SHORT,
    ],
};
