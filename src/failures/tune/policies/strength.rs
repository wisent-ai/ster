//! How choosing the strength a steering artifact is added at fails
//! (src/tune/objective/dpo/strength.rs): the strengths to measure and the
//! settings they are measured with, checked before any pair is scored.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;
use super::run::RUN;

pub static STRENGTHS_NONE: Cause = Cause {
    id: "strengths-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the strength to add the steering artifact at",
    cause: "strength selection needs at least one strength to measure",
    observed: "strengths: []; --batch-size {batch}; --max-sequence {max_sequence}",
    remedy: "name the strengths to measure, comma-separated, such as --strengths 0.5,1,2",
    test: "tests/tuning/scales/strength.sh::no_strength_is_refused",
};

pub static STRENGTH_NOT_FINITE: Cause = Cause {
    id: "strength-not-finite",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the strength to add the steering artifact at",
    cause: "strength selection needs finite strengths, not {strength}",
    observed: "--strengths {strengths}",
    remedy: "give --strengths finite numbers only, such as 0.5,1,2",
    test: "tests/tuning/scales/strength.sh::a_strength_that_is_not_finite_is_refused",
};

pub static SETTING_MISSING: Cause = Cause {
    id: "setting-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the strength to add the steering artifact at",
    cause: "--strengths needs --{flag}; Ster assumes none",
    observed: "--strengths given without --{flag}",
    remedy: "give --{flag} beside --strengths",
    test: "tests/tuning/scales/strength.sh::strengths_without_a_batch_are_refused",
};

pub static SETTING_ZERO: Cause = Cause {
    id: "setting-zero",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the strength to add the steering artifact at",
    cause: "strength selection requires {name} of at least one",
    observed: "--{flag} 0",
    remedy: "give --{flag} at least 1",
    test: "tests/tuning/scales/strength.sh::a_batch_of_zero_is_refused",
};

/// Measuring a steering artifact at each named strength and choosing one.
pub static STRENGTH: Operation = Operation {
    id: "ster.tune.strength",
    entries: &["ster vector evaluate --strengths", "ster request vector/evaluate"],
    uses: &[&RUN],
    summary: "measure how far each named strength moves the model's log-probability of every pair's positive side over its negative side, and choose the one that orders the most pairs",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &STRENGTHS_NONE,
        &STRENGTH_NOT_FINITE,
        &SETTING_MISSING,
        &SETTING_ZERO,
    ],
};
