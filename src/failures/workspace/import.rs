//! How taking a pair set into the workspace fails: naming it, finding the
//! name or the destination taken, and committing the copy.

use wisent_errors::{Cause, Code};

pub static NAME_INVALID: Cause = Cause {
    id: "name-invalid",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "naming the imported set {name}",
    cause: "pair-set name must start with an ASCII letter or digit and contain only letters, digits, dots, underscores, or hyphens",
    observed: "--name {name}",
    remedy: "pass a --name such as my-pairs",
    test: "tests/workspace/failures.sh::an_invalid_name_is_refused",
};

pub static NAME_UNDERIVABLE: Cause = Cause {
    id: "name-underivable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "deriving a pair-set id from {source}",
    cause: "{source} has no ASCII letter or digit in its file name, so no pair-set id can be derived from it",
    observed: "file name: {file}",
    remedy: "name the set with --name, such as --name my-pairs",
    test: "tests/workspace/failures.sh::a_file_name_without_letters_needs_a_name",
};

pub static NAME_FILELESS: Cause = Cause {
    id: "name-fileless",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "deriving a pair-set id from {source}",
    cause: "{source} ends in no file name (it is a root or ends in ..), so no pair-set id can be derived from it",
    observed: "file name: {file}",
    remedy: "name the pair-set file itself, or name the set with --name",
    test: "tests/workspace/failures.sh::a_source_without_a_file_name_needs_a_name",
};

pub static NAME_NOT_UTF8: Cause = Cause {
    id: "name-not-utf8",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "deriving a pair-set id from {source}",
    cause: "{source} has a file name that is not UTF-8, and a pair-set id is text",
    observed: "file name, its bytes that are not UTF-8 replaced: {file}",
    remedy: "name the set with --name, or rename the file to a UTF-8 name",
    test: "tests/workspace/failures.sh::a_file_name_that_is_not_utf8_needs_a_name",
};

pub static NAME_CONFLICT: Cause = Cause {
    id: "name-conflict",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "importing {source} as {id}",
    cause: "pair-set name conflicts with existing content",
    observed: "{id} holds {path}, a different set",
    remedy: "choose another --name to preserve {path}",
    test: "tests/workspace/failures.sh::a_taken_name_is_refused_and_the_first_set_kept",
};

pub static DESTINATION_EXISTS: Cause = Cause {
    id: "destination-exists",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "importing {source} as {id}",
    cause: "the destination already exists outside the Ster workspace index; it was not replaced",
    observed: "file: {path}",
    remedy: "move {path} out of the workspace, or import the set under another --name",
    test: "tests/workspace/failures.sh::a_file_outside_the_index_is_not_replaced",
};

pub static DESTINATION_APPEARED: Cause = Cause {
    id: "destination-appeared",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "importing {source} as {id}",
    cause: "the destination appeared while the import was being committed; it was not replaced",
    observed: "file: {path}; {left}",
    remedy: "move {path} out of the workspace, or import the set under another --name",
    test: "tests/workspace/failures.sh::a_destination_appearing_mid_import_is_not_replaced",
};

pub static STAGING_LEFT: Cause = Cause {
    id: "staging-left",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "staging the import of {source} as {id}",
    cause: "a staging file of an earlier import that did not finish is in the way",
    observed: "staging file: {path}",
    remedy: "remove {path}: no workspace entry records it",
    test: "tests/workspace/failures.sh::a_leftover_staging_file_is_refused",
};

pub static LINK_UNSUPPORTED: Cause = Cause {
    id: "link-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "committing the import of {source} as {id}",
    cause: "the file system holding {directory} does not make hard links, and an import is committed by linking its staged copy into place so no reader sees half of it",
    observed: "{path}: not created; {left}; the system answered: {error}",
    remedy: "keep the workspace on a file system that makes hard links (point XDG_DATA_HOME at one), then import again",
    test: "tests/workspace/failures.sh::a_commit_the_filesystem_refuses_is_rolled_back",
};
