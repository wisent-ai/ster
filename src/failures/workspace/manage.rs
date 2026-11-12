//! How choosing the active set, taking a set out, and reading the active
//! set for a command run without --pairs fail.

use wisent_errors::{Cause, Code};

pub static ALREADY_ACTIVE: Cause = Cause {
    id: "already-active",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "selecting pair set {id}",
    cause: "pair set {id} is already the active one",
    observed: "active pair set: {id}",
    remedy: "nothing needs doing: commands run without --pairs already read {id}; name another set to change it",
    test: "tests/workspace/failures.sh::selecting_the_active_set_is_refused",
};

pub static NO_ACTIVE: Cause = Cause {
    id: "no-active-set",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "choosing the pair set to read",
    cause: "no active Ster pair set",
    observed: "the workspace holds {held}",
    remedy: "pass --pairs, select an imported set with `ster workspace select <id>`, or import one with `ster workspace import`",
    test: "tests/workspace/failures.sh::a_run_without_pairs_and_no_active_set_is_refused",
};

pub static ACTIVE_UNKNOWN: Cause = Cause {
    id: "active-unknown",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "choosing the pair set to read",
    cause: "the Ster workspace names {id} active but holds no pair set by that name",
    observed: "it holds {held}",
    remedy: "select a set it holds with `ster workspace select <id>`, or pass --pairs",
    test: "tests/workspace/failures.sh::an_active_set_the_workspace_does_not_hold_is_refused",
};

pub static ACTIVE_COPY_MISSING: Cause = Cause {
    id: "active-copy-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading the active pair set {id}",
    cause: "active Ster pair set is missing: {path}",
    observed: "the workspace records {id} at {path}",
    remedy: "import it again with `ster workspace import`, select another with `ster workspace select <id>`, or take it out with `ster workspace remove {id}`",
    test: "tests/workspace/failures.sh::an_active_set_whose_copy_is_gone_is_refused",
};
