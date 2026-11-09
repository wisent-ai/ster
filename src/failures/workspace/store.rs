//! How the workspace document and its directory fail, and how naming a set
//! the workspace holds fails: the layers every workspace functionality uses.

use wisent_errors::{Cause, Code};

pub static HOME_UNSET: Cause = Cause {
    id: "home-unset",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "locating the Ster workspace",
    cause: "neither XDG_DATA_HOME nor HOME is set",
    observed: "XDG_DATA_HOME and HOME are unset or empty",
    remedy: "set XDG_DATA_HOME, or HOME, to the directory the workspace lives under: $XDG_DATA_HOME/ster, else ~/.local/share/ster",
    test: "tests/workspace/failures.sh::a_workspace_without_a_home_is_refused",
};

pub static UNREADABLE: Cause = Cause {
    id: "workspace-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the Ster workspace {path}",
    cause: "the file cannot be read: {error}",
    observed: "path: {path}",
    remedy: "make {path} a file this account can read",
    test: "tests/workspace/failures.sh::an_unreadable_workspace_is_refused",
};

pub static MALFORMED: Cause = Cause {
    id: "workspace-malformed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the Ster workspace {path}",
    cause: "it is not a Ster workspace document",
    observed: "invalid Ster workspace JSON in {path}: {error}",
    remedy: "restore {path} from a copy, or move it aside and import the pair sets again with `ster workspace import`",
    test: "tests/workspace/failures.sh::a_workspace_that_is_not_its_document_is_refused",
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

pub static DIRECTORY: Cause = Cause {
    id: "workspace-directory",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "creating the Ster workspace directory {directory}",
    cause: "{error}",
    observed: "the workspace lives under {root}",
    remedy: "point XDG_DATA_HOME, or HOME, at a directory this account can write",
    test: "tests/workspace/failures.sh::a_workspace_under_a_file_is_refused",
};

pub static WRITE: Cause = Cause {
    id: "workspace-unwritable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "writing the Ster workspace {path}",
    cause: "{step} failed: {error}",
    observed: "{left}",
    remedy: "make {directory} writable for this account and run the command again",
    test: "tests/workspace/failures.sh::a_read_only_workspace_is_refused_and_the_import_undone",
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
