//! How authoring a pair set by hand fails: adding, removing and editing one
//! pair, the editor's save, and merging several sets into one.

use wisent_errors::{Cause, Code, Operation};

use super::{DOCS, FILE};

pub static INDEX_OUTSIDE: Cause = Cause {
    id: "index-outside",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "finding pair {index} of {path}",
    cause: "pair index {index} is outside the set: {path} holds {count} pair(s), indexed from 0",
    observed: "pairs: {count}, the last at index {last}",
    remedy: "name an index from 0 to {last}; the pairs of {path} are numbered in the order the file lists them",
    test: "tests/pairs/author.sh::an_index_past_the_end_is_refused",
};

pub static SIDE_EMPTY: Cause = Cause {
    id: "side-empty",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "writing the {side} side of a pair in {path}",
    cause: "--{side} is empty or only whitespace, and every pair needs text on both sides",
    observed: "--{side} \"{value}\"",
    remedy: "give --{side} the text of that side",
    test: "tests/pairs/author.sh::an_empty_side_is_refused_before_the_set_is_read",
};

pub static NOTHING_TO_CHANGE: Cause = Cause {
    id: "nothing-to-change",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "editing pair {index} of {path}",
    cause: "pairs edit changes nothing without --positive, --negative or --trait",
    observed: "none of --positive, --negative or --trait is given; {path} ({state}) is left as it is",
    remedy: "name what changes: --positive <text>, --negative <text> or --trait <name>",
    test: "tests/pairs/author.sh::an_edit_naming_no_change_is_refused",
};

pub static TOO_FEW_SOURCES: Cause = Cause {
    id: "too-few-sources",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "merging pair sets",
    cause: "pairs merge needs at least two pair sets and got {count}",
    observed: "pair sets named: {sources}",
    remedy: "name every set to merge with its own --pairs, two or more: one set is already itself",
    test: "tests/pairs/merge.sh::fewer_than_two_sets_are_refused_before_any_is_read",
};

pub static TRAIT_BLANK: Cause = Cause {
    id: "trait-blank",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "merging pair sets",
    cause: "pairs merge needs the trait name the merged set is fitted for",
    observed: "--trait \"{trait}\"",
    remedy: "give --trait the name of the trait every merged set expresses",
    test: "tests/pairs/merge.sh::a_blank_trait_is_refused_before_any_set_is_read",
};

/// Finding one pair by the index `remove` and `edit` take.
pub static INDEX: Operation = Operation {
    id: "ster.pairs.index",
    entries: &[],
    uses: &[],
    summary: "find one pair of a set by the zero-based index `ster pairs remove` and `ster pairs edit` take",
    docs: DOCS,
    warnings: &[],
    causes: &[&INDEX_OUTSIDE],
};

/// Checking the text given for a side before any set is read.
pub static SIDE: Operation = Operation {
    id: "ster.pairs.side",
    entries: &[],
    uses: &[],
    summary: "check the text given for one side of a pair before any set is read",
    docs: DOCS,
    warnings: &[],
    causes: &[&SIDE_EMPTY],
};

/// Appending one pair.
pub static ADD: Operation = Operation {
    id: "ster.pairs.add",
    entries: &["ster pairs add"],
    uses: &[&SIDE, &FILE],
    summary: "append one pair to a set, creating the file when nothing is there",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// Taking one pair out.
pub static REMOVE: Operation = Operation {
    id: "ster.pairs.remove",
    entries: &["ster pairs remove"],
    uses: &[&INDEX, &FILE],
    summary: "take one pair out of a set by its index; the last pair stays, since a set without pairs is refused",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// Changing one pair in place.
pub static EDIT: Operation = Operation {
    id: "ster.pairs.edit",
    entries: &["ster pairs edit"],
    uses: &[&SIDE, &INDEX, &FILE],
    summary: "change one pair's sides, or the set's trait name, in place",
    docs: DOCS,
    warnings: &[],
    causes: &[&NOTHING_TO_CHANGE],
};

/// The desktop editor's write.
pub static SAVE: Operation = Operation {
    id: "ster.pairs.save",
    entries: &["ster request pairs/save"],
    uses: &[&FILE],
    summary: "write the pair set the desktop editor holds, checked as every loaded set is",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// Several sets written as one.
pub static MERGE: Operation = Operation {
    id: "ster.pairs.merge",
    entries: &["ster pairs merge", "ster request pairs/merge"],
    uses: &[&FILE],
    summary: "write one pair set out of several, every pair of each in the order the sets are named, under one trait name",
    docs: DOCS,
    warnings: &[],
    causes: &[&TOO_FEW_SOURCES, &TRAIT_BLANK],
};
