//! How Ster's pair sets fail: reading, checking and writing one file (the
//! layer every functionality that takes a pair set uses: `PairSet::load`,
//! `validate`, `save`), authoring one pair at a time, merging, auditing, and
//! importing a benchmark export. A file that cannot be read or written, or
//! is not JSON of a pair set's form, is the `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::{file, refuse};

/// What a pair set is, for the `ster.file` refusals that name it.
pub const FORM: &str = "the JSON `ster pairs add`, `ster pairs import` or `ster pairs synthesize` writes, {\"trait_name\": \"<trait>\", \"pairs\": [{\"positive\": \"<text>\", \"negative\": \"<text>\"}]} as docs/examples/pairs.json shows";

pub mod author;
#[path = "io/export.rs"]
pub mod export;
#[path = "io/import.rs"]
pub mod import;
pub mod inspect;
#[path = "io/rows.rs"]
pub mod rows;
pub mod synthesize;

/// docs/guide/errors/pairs.md, where the file layer, authoring, merging and
/// auditing are listed.
const DOCS: &str = page!("pairs");

pub static MISSING: Cause = Cause {
    id: "pair-set-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading pair set {path}",
    cause: "no file is there: nothing at {path} is the pair set named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name an existing pair-set file (`ls -ln {parent}` shows what that directory holds), or write one with `ster pairs add --pairs {path} --positive <text> --negative <text>`",
    test: "tests/pairs/imports/file.sh::a_missing_pair_set_is_refused",
};

pub static EMPTY: Cause = Cause {
    id: "pair-set-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking pair set {label}",
    cause: "pair set {label} contains no pairs",
    observed: "pairs: 0; trait_name: \"{trait}\"",
    remedy: "add a pair with `ster pairs add --pairs <file> --positive <text> --negative <text>`, or name a set that holds pairs",
    test: "tests/pairs/imports/file.sh::removing_the_last_pair_is_refused_and_the_file_kept",
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
    test: "tests/pairs/imports/file.sh::a_pair_with_an_empty_side_is_refused",
};

/// Reading, checking and writing one pair-set file.
pub static FILE: Operation = Operation {
    id: "ster.pairs.file",
    entries: &[],
    uses: &[&file::FILE],
    summary: "read a pair-set file, check that every pair has both sides, and write one back",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &MISSING,
        &EMPTY,
        &BLANK,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[
    &FILE,
    &author::INDEX,
    &author::SIDE,
    &author::ADD,
    &author::REMOVE,
    &author::EDIT,
    &author::SAVE,
    &author::MERGE,
    &inspect::DEDUPE,
    &inspect::INSPECT,
    &rows::ROWS,
    &import::IMPORT,
    &export::EXPORT,
    &synthesize::SYNTHESIZE,
];
