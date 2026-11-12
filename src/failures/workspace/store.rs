//! How the workspace document and its directory fail, and how naming a set
//! the workspace holds fails: the layers every workspace functionality uses.
//! Reading, writing and replacing the document or its directory failing is
//! the `ster.file` layer's cause.

use wisent_errors::{Cause, Code};

/// What the workspace document is, for the `ster.file` refusals that name it.
pub const FORM: &str = "the record Ster writes itself (restore it from a copy, or move it aside and import the pair sets again with `ster workspace import`)";

pub static HOME_UNSET: Cause = Cause {
    id: "home-unset",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "locating the Ster workspace",
    cause: "neither XDG_DATA_HOME nor HOME is set",
    observed: "XDG_DATA_HOME is {xdg}; HOME is {home}",
    remedy: "set XDG_DATA_HOME, or HOME, to the directory the workspace lives under: $XDG_DATA_HOME/ster, else ~/.local/share/ster",
    test: "tests/workspace/failures.sh::a_workspace_without_a_home_is_refused",
};

pub static SCHEMA: Cause = Cause {
    id: "workspace-schema",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the Ster workspace {path}",
    cause: "unsupported Ster workspace schema {found}; this ster reads {expected}",
    observed: "schema: {found}",
    remedy: "read it with the ster that wrote it, or move {path} aside and import the pair sets again with `ster workspace import`",
    test: "tests/workspace/failures.sh::a_workspace_of_another_schema_is_refused",
};

pub static SET_UNKNOWN: Cause = Cause {
    id: "set-unknown",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "finding pair set {id}",
    cause: "the Ster workspace holds no pair set named {id}; it holds {held}",
    observed: "workspace: {path}",
    remedy: "name a set the workspace holds, as `ster workspace show` prints them, or import it with `ster workspace import`",
    test: "tests/workspace/failures.sh::an_unknown_set_is_refused",
};
