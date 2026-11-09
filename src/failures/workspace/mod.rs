//! How Ster's local workspace fails: the pair sets a person imported, which
//! one is active, and the document that records them.

use wisent_errors::Operation;

use super::pairs::FILE;

/// docs/guide/errors/workspace.md, where these functionalities are listed.
const DOCS: &str = page!("workspace");

pub mod import;
pub mod manage;
pub mod store;

/// Locating, reading and writing the workspace document and its directory.
pub static STORE: Operation = Operation {
    id: "ster.workspace.store",
    entries: &[],
    uses: &[],
    summary: "locate the workspace under XDG_DATA_HOME or HOME, read its document, and write it so a reader never sees half of it",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &store::HOME_UNSET,
        &store::UNREADABLE,
        &store::MALFORMED,
        &store::SCHEMA,
        &store::DIRECTORY,
        &store::WRITE,
    ],
};

/// Naming one pair set the workspace holds.
pub static SET: Operation = Operation {
    id: "ster.workspace.set",
    entries: &[],
    uses: &[&STORE],
    summary: "find an imported pair set by the id `ster workspace show` prints",
    docs: DOCS,
    warnings: &[],
    causes: &[&store::SET_UNKNOWN],
};

/// Taking a pair-set file into the workspace and making it active.
pub static IMPORT: Operation = Operation {
    id: "ster.workspace.import",
    entries: &[
        "ster workspace import",
        "ster onboarding --import-pairs",
        "ster request workspace/import",
    ],
    uses: &[&FILE, &STORE],
    summary: "validate a pair-set file, copy it into the workspace under its id, and make it the active set; a repeat of the same content is the same import",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &import::NAME_INVALID,
        &import::NAME_UNDERIVABLE,
        &import::NAME_CONFLICT,
        &import::DESTINATION_EXISTS,
        &import::DESTINATION_APPEARED,
        &import::STAGING_LEFT,
        &import::COMMIT_FAILED,
        &import::STAGING_UNREMOVABLE,
    ],
};

/// Printing the imported sets and the active one.
pub static SHOW: Operation = Operation {
    id: "ster.workspace.show",
    entries: &["ster workspace show", "ster request workspace/show"],
    uses: &[&STORE],
    summary: "print every imported pair set and which one is active",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// Making an imported set the active one.
pub static SELECT: Operation = Operation {
    id: "ster.workspace.select",
    entries: &["ster workspace select", "ster request workspace/select"],
    uses: &[&SET, &STORE],
    summary: "make an imported pair set the one every command run without --pairs reads",
    docs: DOCS,
    warnings: &[],
    causes: &[&manage::ALREADY_ACTIVE],
};

/// Taking an imported set out and deleting the workspace's copy.
pub static REMOVE: Operation = Operation {
    id: "ster.workspace.remove",
    entries: &["ster workspace remove", "ster request workspace/remove"],
    uses: &[&SET, &STORE],
    summary: "delete the workspace's copy of an imported pair set and take it out of the workspace; the source file is not touched",
    docs: DOCS,
    warnings: &[],
    causes: &[&manage::COPY_UNDELETABLE],
};

/// Reading the active set for a command given no --pairs.
pub static ACTIVE: Operation = Operation {
    id: "ster.workspace.active",
    entries: &[
        "ster pairs inspect",
        "ster vector train",
        "ster vector optimize",
        "ster vector evaluate",
        "ster vector curve",
        "ster vector project",
        "ster tune dpo",
        "ster tune bipo",
        "ster tune reward",
    ],
    uses: &[&STORE],
    summary: "name the active imported pair set's copy for a command run without --pairs",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &manage::NO_ACTIVE,
        &manage::ACTIVE_UNKNOWN,
        &manage::ACTIVE_COPY_MISSING,
    ],
};

pub(super) static OPERATIONS: &[&Operation] =
    &[&STORE, &SET, &IMPORT, &SHOW, &SELECT, &REMOVE, &ACTIVE];
