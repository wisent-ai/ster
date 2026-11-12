//! How the GitHub star request fails (src/cli/star.rs): the `star-request`
//! journey the shared onboarding runtime (`wisent-onboarding-client`) runs
//! and keeps in `$XDG_STATE_HOME/ster/onboarding.json`, read by `ster star`
//! and `ster request star/*`, and counted after a run of a model completes.
//! The runtime answers one of four kinds of `JourneyError`, each a sentence;
//! the kind decides the cause and the sentence is quoted.

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/onboarding.md, where this functionality is listed.
const DOCS: &str = page!("onboarding");

const STEP: &str = "{step} the GitHub star request kept in {file}";

pub static STORAGE: Cause = Cause {
    id: "journey-storage",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: STEP,
    cause: "the onboarding runtime could not keep the request's journey state: its file or directory cannot be created, opened, locked, read or written by this account, it holds no journey state, or neither XDG_STATE_HOME nor HOME names where it lives",
    observed: "{file} ({state}); its directory {directory} ({directory_state}); XDG_STATE_HOME {xdg}; HOME {home}; USER {user}; the runtime answered: {said}",
    remedy: "give this account write access to {directory}, or set XDG_STATE_HOME to a directory it may write; `ster star reset` sets aside a file that holds no journey state",
    test: "tests/onboarding/star.sh::a_state_directory_that_cannot_be_made_is_refused",
};

pub static INVALID: Cause = Cause {
    id: "journey-invalid",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: STEP,
    cause: "the onboarding runtime refused the star-request journey: the journey it carries or the progress stored for it names a screen, stage or repository the runtime does not read",
    observed: "{file} ({state}); repository {repository}; the runtime answered: {said}",
    remedy: "`ster star reset` counts from zero with the journey this ster carries; if it refuses too, install a ster release built with a current wisent-onboarding-client",
    test: "tests/onboarding/star.sh::a_journey_the_runtime_refuses_is_refused",
};

pub static TRANSPORT: Cause = Cause {
    id: "journey-transport",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: STEP,
    cause: "the onboarding runtime's transport did not answer, which the offline transport the star request uses never does",
    observed: "{file} ({state}); the runtime answered: {said}",
    remedy: "run the command again; if it refuses again, install a ster release built with a current wisent-onboarding-client",
    test: "tests/onboarding/star.sh::a_transport_that_does_not_answer_is_refused",
};

pub static NOT_STARTED: Cause = Cause {
    id: "journey-not-started",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: STEP,
    cause: "the onboarding runtime holds no progress of the star-request journey right after opening it, so there is no stage to read",
    observed: "{file} ({state}); the runtime answered: {said}",
    remedy: "`ster star reset` starts the journey again from zero",
    test: "tests/onboarding/star.sh::a_journey_without_progress_is_refused",
};

/// Reading, dismissing, resetting and counting the GitHub star request.
pub static STAR: Operation = Operation {
    id: "ster.star",
    entries: &[
        "ster star status",
        "ster star dismiss",
        "ster star reset",
        "ster request star/status",
        "ster request star/dismiss",
        "ster request star/reset",
    ],
    uses: &[],
    summary: "read, dismiss or reset the GitHub star request the shared onboarding runtime keeps per user, and count each completed run of a model toward it",
    docs: DOCS,
    warnings: &[],
    causes: &[&STORAGE, &INVALID, &TRANSPORT, &NOT_STARTED],
};

pub(super) static OPERATIONS: &[&Operation] = &[&STAR];
