//! How walking an operator through Ster's first use fails (src/cli/onboarding):
//! locating, reading and writing the progress it keeps between runs. The
//! journey itself is compiled into the binary and checked when it is built;
//! the state file failing to be read, parsed or written is the `ster.file`
//! layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::file::FILE;
use super::workspace::IMPORT;

/// docs/guide/errors/onboarding.md, where this functionality is listed.
const DOCS: &str = page!("onboarding");

/// What the onboarding state is, for the `ster.file` refusals that name it.
pub const FORM: &str = "the JSON `ster onboarding` writes there (`ster onboarding --reset` writes it anew)";

pub static HOME_UNSET: Cause = Cause {
    id: "onboarding-home-unset",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "locating the onboarding state",
    cause: "neither XDG_STATE_HOME nor HOME is set, so there is no directory of this account to keep the walkthrough's progress in",
    observed: "XDG_STATE_HOME: {xdg}; HOME: {home}",
    remedy: "set XDG_STATE_HOME, or HOME, to the directory the progress lives under: $XDG_STATE_HOME/ster, else ~/.local/state/ster",
    test: "tests/onboarding/state.sh::a_walkthrough_without_a_home_is_refused",
};

pub static STATE_FOREIGN: Cause = Cause {
    id: "onboarding-state-foreign",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the onboarding state {path}",
    cause: "the state records another journey than the one this Ster walks: its {field} is {found}, and this Ster's is {expected}",
    observed: "path: {path}; {field}: {found}",
    remedy: "run `ster onboarding --reset` to start this journey anew; the recorded progress belongs to another release",
    test: "tests/onboarding/state.sh::a_state_of_another_journey_is_refused",
};

pub static SCREEN_UNKNOWN: Cause = Cause {
    id: "onboarding-screen-unknown",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the onboarding state {path}",
    cause: "the state stands at screen {screen}, which this journey does not hold, so the walkthrough cannot go on from it",
    observed: "path: {path}; current_screen_id: {screen}; the journey's screens: {screens}",
    remedy: "run `ster onboarding --reset` to start the journey from its first screen",
    test: "tests/onboarding/state.sh::a_state_at_an_unknown_screen_is_refused",
};

/// Walking through first use and recording its one success.
pub static ONBOARDING: Operation = Operation {
    id: "ster.onboarding",
    entries: &["ster onboarding"],
    uses: &[&FILE, &IMPORT],
    summary: "show the first-use walkthrough from the screen the operator reached, keep the progress in $XDG_STATE_HOME/ster/first-use.json, and record its completion once a pair set is imported",
    docs: DOCS,
    warnings: &[],
    causes: &[&HOME_UNSET, &STATE_FOREIGN, &SCREEN_UNKNOWN],
};

pub(super) static OPERATIONS: &[&Operation] = &[&ONBOARDING];
