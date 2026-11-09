//! How reading, checking and writing a pair-set file fails: the layer every
//! functionality that takes a pair set uses (`PairSet::load`, `validate`,
//! `save`).

use wisent_errors::{Cause, Code, Operation};

use super::refuse;

/// docs/guide/errors/pairs.md, where this functionality is listed.
const DOCS: &str = page!("pairs");

/// The refusal a pair-set file that cannot be read meets: nothing there, or
/// something there this account cannot read as a file.
pub fn read_refusal(path: &str, error: &std::io::Error) -> anyhow::Error {
    let cause = if error.kind() == std::io::ErrorKind::NotFound {
        &MISSING
    } else {
        &UNREADABLE
    };
    refuse(cause, &[("path", path), ("error", &error.to_string())])
}

pub static MISSING: Cause = Cause {
    id: "pair-set-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading pair set {path}",
    cause: "no file is there: {error}",
    observed: "path: {path}",
    remedy: "name an existing pair-set file, or write one with `ster pairs add --pairs {path} --positive <text> --negative <text>`",
    test: "tests/pairs/file.sh::a_missing_pair_set_is_refused",
};

pub static UNREADABLE: Cause = Cause {
    id: "pair-set-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading pair set {path}",
    cause: "the file cannot be read: {error}",
    observed: "path: {path}",
    remedy: "name a pair-set file this account can read",
    test: "tests/pairs/file.sh::a_directory_named_as_a_pair_set_is_refused",
};

pub static MALFORMED: Cause = Cause {
    id: "pair-set-malformed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading pair set {path}",
    cause: "the file is not a pair-set document",
    observed: "invalid pair set JSON in {path}: {error}",
    remedy: "write it as {\"trait_name\": \"<trait>\", \"pairs\": [{\"positive\": \"<text>\", \"negative\": \"<text>\"}]}, as docs/examples/pairs.json is",
    test: "tests/pairs/file.sh::a_file_that_is_not_a_pair_set_is_refused",
};

pub static EMPTY: Cause = Cause {
    id: "pair-set-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking pair set {label}",
    cause: "pair set {label} contains no pairs",
    observed: "pairs: 0",
    remedy: "add a pair with `ster pairs add --pairs <file> --positive <text> --negative <text>`, or name a set that holds pairs",
    test: "tests/pairs/file.sh::removing_the_last_pair_is_refused_and_the_file_kept",
};

pub static BLANK: Cause = Cause {
    id: "pair-set-blank-prompt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking pair set {label}",
    cause: "pair set {label} contains an empty positive or negative prompt",
    observed: "pair {index} has an empty {side}",
    remedy: "write that side with `ster pairs edit --pairs <file> --index {index} --{side} <text>`, or take the pair out with `ster pairs remove --pairs <file> --index {index}`",
    test: "tests/pairs/file.sh::a_pair_with_an_empty_side_is_refused",
};

pub static DIRECTORY: Cause = Cause {
    id: "pair-set-directory",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "writing pair set {path}",
    cause: "its directory {directory} cannot be created: {error}",
    observed: "path: {path}",
    remedy: "name a path whose directory this account can create, or create {directory} first",
    test: "tests/pairs/file.sh::a_set_under_a_file_is_refused",
};

pub static UNWRITABLE: Cause = Cause {
    id: "pair-set-unwritable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "writing pair set {path}",
    cause: "the file cannot be written: {error}",
    observed: "path: {path}",
    remedy: "make its directory writable for this account, or name another path",
    test: "tests/pairs/file.sh::a_set_in_a_read_only_directory_is_refused",
};

/// Reading, checking and writing one pair-set file.
pub static FILE: Operation = Operation {
    id: "ster.pairs.file",
    entries: &[],
    uses: &[],
    summary: "read a pair-set file, check that every pair has both sides, and write one back",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &MISSING,
        &UNREADABLE,
        &MALFORMED,
        &EMPTY,
        &BLANK,
        &DIRECTORY,
        &UNWRITABLE,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&FILE];
