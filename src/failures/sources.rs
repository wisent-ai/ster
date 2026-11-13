//! How `ster sources` fails: the data other tools keep that Ster reads, listed one adapter
//! each, and one file read as its adapter's command would read it. A file that is not the
//! adapter's source is answered as `why_not`, so the only refusal is an adapter Ster does
//! not have.

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

/// Listing the sources Ster reads and discovering one file.
pub static SOURCES: Operation = Operation {
    id: "ster.sources",
    entries: &["ster sources"],
    uses: &[],
    summary: "list the data other tools keep that Ster reads, one adapter each (IMP-STER-<NN>), and say how many examples or pairs one file gives",
    docs: DOCS,
    causes: &[&ADAPTER_UNKNOWN],
    warnings: &[],
};

pub(super) const OPERATIONS: &[&Operation] = &[&SOURCES];
