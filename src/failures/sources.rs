//! How `ster sources` fails: the data other tools keep that Ster reads, listed one adapter
//! each, and one file read as its adapter's command would read it. A file that is not the
//! adapter's source is answered as `why_not`, so the refusals are an adapter Ster does not
//! have, an adapter named by the requirement number it was once listed under, and an
//! answer to `sources share` that cannot be kept.

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/sources.md, where this functionality is listed.
const DOCS: &str = page!("sources");

pub static ADAPTER_UNKNOWN: Cause = Cause {
    id: "sources-adapter-unknown",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "discovering {path} as {adapter}",
    cause: "{adapter} is not an adapter of ster",
    observed: "its adapters are {known}",
    remedy: "name one of the adapters `ster sources` lists",
    test: "tests/sources/sources.sh::an_unknown_adapter_is_refused",
};

/// An adapter named by the import plan's requirement number it was listed under before
/// adapters had names: no adapter answers to the number, and the refusal names the name.
pub static ADAPTER_RENAMED: Cause = Cause {
    id: "sources-adapter-renamed",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "discovering {path} as {adapter}",
    cause: "{adapter} is the import plan's requirement number of an adapter, and ster's adapters are named, not numbered",
    observed: "{number} stands for the adapter named {name}",
    remedy: "name it {name}: `ster sources discover {name} --path {path}`",
    test: "tests/sources/sources.sh::an_adapter_named_by_its_requirement_number_is_refused_naming_its_name",
};

/// `ster sources share yes|no` keeps its answer in a file; one it cannot write is refused.
pub static SHARE_UNKEPT: Cause = Cause {
    id: "sources-share-unkept",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "keeping whether discoveries report their counts",
    cause: "{why}",
    observed: "the answer was not kept",
    remedy: "set XDG_CONFIG_HOME to a directory this user can write, then answer again",
    test: "tests/sources/sources.sh::an_answer_to_sharing_that_cannot_be_kept_is_refused",
};

/// Listing the sources Ster reads and discovering one file.
pub static SOURCES: Operation = Operation {
    id: "ster.sources",
    entries: &["ster sources", "ster sources discover", "ster sources share"],
    uses: &[],
    summary: "list the data other tools keep that Ster reads, one adapter each, named for its source, and say how many examples or pairs one file gives",
    docs: DOCS,
    causes: &[&ADAPTER_UNKNOWN, &ADAPTER_RENAMED, &SHARE_UNKEPT],
    warnings: &[],
};

pub(super) const OPERATIONS: &[&Operation] = &[&SOURCES];
