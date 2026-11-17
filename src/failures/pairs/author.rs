//! How authoring a pair set by hand fails: adding, removing and editing one
//! pair, the editor's save, merging several sets into one, and drawing seeded
//! sets out of one (splitting it, sampling it).

use wisent_errors::{Cause, Code, Operation};

use super::super::workflow::holdout::HOLDOUT;
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

pub static OUTPUT_EXISTS: Cause = Cause {
    id: "output-exists",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing --{flag} {path}",
    cause: "{path} already holds a file, and a drawn pair set is written only where none is; nothing was written",
    observed: "--{flag} {path} ({state})",
    remedy: "give --{flag} a path where no file is, or move {path} away first",
    test: "tests/pairs/draw.sh::an_output_that_exists_is_refused",
};

/// Checking that a drawn set's output names no file yet.
pub static OUTPUT: Operation = Operation {
    id: "ster.pairs.output",
    entries: &[],
    uses: &[],
    summary: "check, before any set is read, that the path a drawn pair set is written to holds no file, so a set already there is never replaced",
    docs: DOCS,
    warnings: &[],
    causes: &[&OUTPUT_EXISTS],
};

pub static OUTPUTS_SAME: Cause = Cause {
    id: "outputs-same",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing where pairs split writes",
    cause: "--train and --test both name {path}, and the two sides are two files",
    observed: "--train {path} --test {path}",
    remedy: "give --train and --test two different paths",
    test: "tests/pairs/draw.sh::one_path_for_both_sides_is_refused",
};

/// A set parted into a training and a held-out set by a seeded shuffle.
pub static SPLIT: Operation = Operation {
    id: "ster.pairs.split",
    entries: &["ster pairs split"],
    uses: &[&OUTPUT, &HOLDOUT, &FILE],
    summary: "part a pair set into a training set and a held-out set by a shuffle the seed fixes, each side keeping the set's order, and write both",
    docs: DOCS,
    warnings: &[],
    causes: &[&OUTPUTS_SAME],
};

pub static COUNT_OUTSIDE: Cause = Cause {
    id: "count-outside",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "drawing {count} pairs from {path}",
    cause: "--count {count} is not between 1 and the {total} pair(s) the set holds, and a sample repeats no pair",
    observed: "--count {count}; {path} holds {total} pair(s)",
    remedy: "give --count a number from 1 to {total}",
    test: "tests/pairs/draw.sh::a_count_past_the_set_is_refused",
};

/// A number of pairs drawn out of a set by a seeded shuffle.
pub static SAMPLE: Operation = Operation {
    id: "ster.pairs.sample",
    entries: &["ster pairs sample"],
    uses: &[&OUTPUT, &FILE],
    summary: "draw a number of pairs out of a pair set by a shuffle the seed fixes, keeping the set's order, and write them",
    docs: DOCS,
    warnings: &[],
    causes: &[&COUNT_OUTSIDE],
};
