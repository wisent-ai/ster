//! Taking a pair set someone else produced into the workspace: what is
//! accepted, what the copy is named, and when a repeat is the same import
//! rather than a new one.

use std::{fs, path::Path};

use anyhow::Result;
use blake2::{Blake2b512, Digest};
use serde::Serialize;
use serde_json::Value;

use crate::PairSet;
use crate::failures::{self, refuse, workspace::import};

use super::name::{derived_name, validate_name};
use super::state::{PairSetEntry, create_directory, load_state, save_state, workspace_root};

/// How much of a pair set's BLAKE2b-512 digest the workspace keeps as its
/// content identity, as every recorded `digest` holds it.
const PAIR_SET_DIGEST_BYTES: usize = 16; // https://github.com/wisent-ai/ster/blob/main/docs/guide/pair-sets/workspace.md "the first 16 bytes are kept"

/// How an import ended, printed as its lower-case name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ImportStatus {
    /// The set was copied into the workspace.
    Imported,
    /// The workspace already holds this exact set under this name.
    Unchanged,
    /// The source is not a pair set Ster reads.
    Rejected,
    /// The name is taken by a different set.
    Conflicting,
}

#[derive(Debug, Serialize)]
pub struct ImportReport {
    pub status: ImportStatus,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair_count: Option<usize>,
    /// The refusal of a rejected or conflicting import, as a person reads it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The same refusal's declared cause, as a machine reads it: its id,
    /// code, exit status and fix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<Value>,
    #[serde(skip)]
    refusal: Option<anyhow::Error>,
}

impl ImportReport {
    /// Whether the workspace now holds the set: copied in, or already there.
    pub fn accepted(&self) -> bool {
        matches!(
            self.status,
            ImportStatus::Imported | ImportStatus::Unchanged
        )
    }

    /// The refusal a rejected or conflicting import ends in, taken out of
    /// the report: every surface prints the report, then ends in it.
    pub fn take_refusal(&mut self) -> Option<anyhow::Error> {
        self.refusal.take()
    }

    fn accepted_as(status: ImportStatus, source: &Path, entry: &PairSetEntry) -> Self {
        Self {
            status,
            source: source.display().to_string(),
            id: Some(entry.id.clone()),
            path: Some(entry.path.display().to_string()),
            pair_count: Some(entry.pair_count),
            reason: None,
            cause: None,
            refusal: None,
        }
    }

    fn refused(
        status: ImportStatus,
        source: String,
        found: Option<(&str, &Path)>,
        pair_count: Option<usize>,
        refusal: anyhow::Error,
    ) -> Self {
        Self {
            status,
            source,
            id: found.map(|(id, _)| id.to_owned()),
            path: found.map(|(_, path)| path.display().to_string()),
            pair_count,
            reason: Some(format!("{refusal:#}")),
            cause: failures::declared(&refusal).map(failures::Declared::summary),
            refusal: Some(refusal),
        }
    }
}

/// Validate and adopt an existing canonical pair-set document.
///
/// The source is parsed through `PairSet::load` before the workspace changes.
/// Accepted bytes are serialized back through `PairSet::save`, so the durable
/// copy is exactly the document Ster's training operations read. A content
/// digest makes a repeated import idempotent, while a reused logical name with
/// different content is reported as a conflict and never overwrites the first
/// set. A rejected or conflicting import is a report carrying its refusal;
/// a failure while writing is an error that says what was undone.
pub fn import_pair_set(source: &Path, requested_name: Option<&str>) -> Result<ImportReport> {
    let pair_set = match PairSet::load(source) {
        Ok(pair_set) => pair_set,
        Err(refusal) => return Ok(rejected(source, None, refusal)),
    };
    let count = pair_set.pairs.len();
    // The load just read it, so only a file taken away since then lands here.
    let source = match fs::canonicalize(source) {
        Ok(source) => source,
        Err(error) => {
            let refusal = failures::file::read::refusal_or(
                &failures::pairs::MISSING,
                "the pair set",
                source,
                &error,
            );
            return Ok(rejected(source, None, refusal));
        }
    };
    let named = match requested_name {
        Some(name) => validate_name(name).map(|()| name.to_owned()),
        None => derived_name(&source),
    };
    let id = match named {
        Ok(id) => id,
        Err(refusal) => return Ok(rejected(&source, Some(count), refusal)),
    };
    let shown = source.display().to_string();
    let conflicting = |path: &Path, refusal: anyhow::Error| -> Result<ImportReport> {
        Ok(ImportReport::refused(
            ImportStatus::Conflicting,
            shown.clone(),
            Some((id.as_str(), path)),
            Some(count),
            refusal,
        ))
    };
    let digest = pair_set_digest(&pair_set)?;
    let mut state = load_state()?;

    if let Some(existing) = state
        .pair_sets
        .iter()
        .find(|entry| entry.digest == digest)
        .cloned()
    {
        state.active_pair_set = Some(existing.id.clone());
        save_state(&state)?;
        return Ok(ImportReport::accepted_as(
            ImportStatus::Unchanged,
            &source,
            &existing,
        ));
    }

    if let Some(existing) = state.pair_sets.iter().find(|entry| entry.id == id) {
        let path = existing.path.display().to_string();
        let refusal = refuse(
            &import::NAME_CONFLICT,
            &[("source", &shown), ("id", &id), ("path", &path)],
        );
        return conflicting(existing.path.as_path(), refusal);
    }

    let root = workspace_root()?;
    let pairs_dir = root.join("pairs");
    create_directory(&root, &pairs_dir)?;
    let destination = pairs_dir.join(format!("{id}.json"));
    let destination_shown = destination.display().to_string();
    if destination.exists() {
        let refusal = refuse(
            &import::DESTINATION_EXISTS,
            &[
                ("source", &shown),
                ("id", &id),
                ("path", &destination_shown),
            ],
        );
        return conflicting(destination.as_path(), refusal);
    }

    let temporary = pairs_dir.join(format!(".{id}.{}.incoming", std::process::id()));
    let staging = temporary.display().to_string();
    if temporary.exists() {
        return Err(refuse(
            &import::STAGING_LEFT,
            &[("source", &shown), ("id", &id), ("path", &staging)],
        ));
    }
    pair_set
        .save(&temporary)
        .map_err(|error| failures::amend(error, format!("it was staging the import of {shown} as {id}")))?;
    if let Err(error) = fs::hard_link(&temporary, &destination) {
        let left = left_behind(&temporary);
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            let error_text = error.to_string();
            let values: [(&str, &str); 6] = [
                ("source", &shown),
                ("id", &id),
                ("path", &destination_shown),
                ("staging", &staging),
                ("error", &error_text),
                ("left", &left),
            ];
            let refusal = refuse(&import::DESTINATION_APPEARED, &values);
            return conflicting(destination.as_path(), refusal);
        }
        if error.kind() == std::io::ErrorKind::Unsupported {
            return Err(refuse(
                &import::LINK_UNSUPPORTED,
                &[
                    ("source", &shown),
                    ("id", &id),
                    ("path", &destination_shown),
                    ("directory", &pairs_dir.display().to_string()),
                    ("error", &error.to_string()),
                    ("left", &left),
                ],
            ));
        }
        let action = format!("linking {staging} into place as");
        return Err(failures::file::write::refusal(
            &action,
            "the pair set's copy",
            &destination,
            &left,
            &error,
        ));
    }
    if let Err(error) = fs::remove_file(&temporary) {
        return Err(failures::file::write::refusal(
            "removing",
            "the import's staging file",
            &temporary,
            &left_behind(&destination),
            &error,
        ));
    }

    let entry = PairSetEntry {
        id: id.clone(),
        digest,
        path: destination.clone(),
        source: source.clone(),
        pair_count: count,
    };
    state.pair_sets.push(entry.clone());
    state.active_pair_set = Some(id.clone());
    if let Err(error) = save_state(&state) {
        let left = left_behind(&destination);
        return Err(failures::amend(error, format!("importing {shown} as {id} was undone: {left}")));
    }
    Ok(ImportReport::accepted_as(
        ImportStatus::Imported,
        &source,
        &entry,
    ))
}

/// The report of an import refused before anything was written: the source
/// is not a pair set Ster reads, or it gives no valid id.
fn rejected(source: &Path, pair_count: Option<usize>, refusal: anyhow::Error) -> ImportReport {
    let shown = source.display().to_string();
    ImportReport::refused(ImportStatus::Rejected, shown, None, pair_count, refusal)
}

fn pair_set_digest(pair_set: &PairSet) -> Result<String> {
    let bytes = crate::json::compact(pair_set).into_bytes();
    let digest = Blake2b512::digest(bytes);
    Ok(digest[..PAIR_SET_DIGEST_BYTES]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Remove `path` while undoing an import, and say how that went: removed,
/// or left behind and why. Unsaid, a committed pair set left unrecorded in
/// pairs/ came back on the next import as a conflict nobody could explain.
fn left_behind(path: &Path) -> String {
    match fs::remove_file(path) {
        Ok(()) => format!("{} was removed", path.display()),
        Err(error) => format!(
            "{} could not be removed and is left behind: {error}",
            path.display()
        ),
    }
}
