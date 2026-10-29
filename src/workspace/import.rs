//! Taking a pair set someone else produced into the workspace: what is
//! accepted, what the copy is named, and when a repeat is the same import
//! rather than a new one.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use blake2::{Blake2b512, Digest};
use serde::Serialize;

use crate::PairSet;

use super::state::{PairSetEntry, load_state, save_state, workspace_root};

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

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub status: ImportStatus,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pair_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl ImportReport {
    /// Whether the workspace now holds the set: copied in, or already there.
    pub fn accepted(&self) -> bool {
        matches!(
            self.status,
            ImportStatus::Imported | ImportStatus::Unchanged
        )
    }
}

/// Validate and adopt an existing canonical pair-set document.
///
/// The source is parsed through `PairSet::load` before the workspace changes.
/// Accepted bytes are serialized back through `PairSet::save`, so the durable
/// copy is exactly the document Ster's training operations read. A content
/// digest makes a repeated import idempotent, while a reused logical name with
/// different content is reported as a conflict and never overwrites the first
/// set.
pub fn import_pair_set(source: &Path, requested_name: Option<&str>) -> Result<ImportReport> {
    let source_display = source.display().to_string();
    let pair_set = match PairSet::load(source) {
        Ok(pair_set) => pair_set,
        Err(error) => {
            return Ok(ImportReport {
                status: ImportStatus::Rejected,
                source: source_display,
                id: None,
                path: None,
                pair_count: None,
                reason: Some(format!("{error:#}")),
            });
        }
    };
    let source = fs::canonicalize(source)
        .with_context(|| format!("failed to resolve pair set {}", source.display()))?;
    let named = match requested_name {
        Some(name) => validate_name(name).map(|()| name.to_owned()),
        None => derived_name(&source),
    };
    let id = match named {
        Ok(id) => id,
        Err(reason) => {
            return Ok(ImportReport {
                status: ImportStatus::Rejected,
                source: source.display().to_string(),
                id: None,
                path: None,
                pair_count: Some(pair_set.pairs.len()),
                reason: Some(reason),
            });
        }
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
        return Ok(ImportReport {
            status: ImportStatus::Unchanged,
            source: source.display().to_string(),
            id: Some(existing.id),
            path: Some(existing.path.display().to_string()),
            pair_count: Some(existing.pair_count),
            reason: None,
        });
    }

    if let Some(existing) = state.pair_sets.iter().find(|entry| entry.id == id) {
        return Ok(ImportReport {
            status: ImportStatus::Conflicting,
            source: source.display().to_string(),
            id: Some(id),
            path: Some(existing.path.display().to_string()),
            pair_count: Some(pair_set.pairs.len()),
            reason: Some(format!(
                "pair-set name conflicts with existing content; choose another --name to preserve {}",
                existing.path.display()
            )),
        });
    }

    let root = workspace_root()?;
    let pairs_dir = root.join("pairs");
    fs::create_dir_all(&pairs_dir).with_context(|| {
        format!(
            "failed to create Ster pair-set directory {}",
            pairs_dir.display()
        )
    })?;
    let destination = pairs_dir.join(format!("{id}.json"));
    if destination.exists() {
        return Ok(ImportReport {
            status: ImportStatus::Conflicting,
            source: source.display().to_string(),
            id: Some(id),
            path: Some(destination.display().to_string()),
            pair_count: Some(pair_set.pairs.len()),
            reason: Some("the destination already exists outside the Ster workspace index; it was not replaced".to_owned()),
        });
    }

    let temporary = pairs_dir.join(format!(".{id}.{}.incoming", std::process::id()));
    if temporary.exists() {
        bail!(
            "Ster import staging path already exists: {}",
            temporary.display()
        );
    }
    pair_set.save(&temporary)?;
    if let Err(error) = fs::hard_link(&temporary, &destination) {
        let staging = left_behind(&temporary);
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            let committed = "the destination appeared while the import was being committed; it was not replaced";
            return Ok(ImportReport {
                status: ImportStatus::Conflicting,
                source: source.display().to_string(),
                id: Some(id),
                path: Some(destination.display().to_string()),
                pair_count: Some(pair_set.pairs.len()),
                reason: Some(match staging {
                    Some(note) => format!("{committed}; {note}"),
                    None => committed.to_owned(),
                }),
            });
        }
        let failure = anyhow::Error::new(error).context(format!(
            "failed to commit imported pair set {}",
            destination.display()
        ));
        return Err(with_note(failure, staging));
    }
    if let Err(error) = fs::remove_file(&temporary) {
        let failure = anyhow::Error::new(error).context(format!(
            "failed to remove import staging file {}",
            temporary.display()
        ));
        return Err(with_note(failure, left_behind(&destination)));
    }

    state.pair_sets.push(PairSetEntry {
        id: id.clone(),
        digest,
        path: destination.clone(),
        source: source.clone(),
        pair_count: pair_set.pairs.len(),
    });
    state.active_pair_set = Some(id.clone());
    if let Err(error) = save_state(&state) {
        return Err(with_note(error, left_behind(&destination)));
    }

    Ok(ImportReport {
        status: ImportStatus::Imported,
        source: source.display().to_string(),
        id: Some(id),
        path: Some(destination.display().to_string()),
        pair_count: Some(pair_set.pairs.len()),
        reason: None,
    })
}

fn pair_set_digest(pair_set: &PairSet) -> Result<String> {
    let bytes = serde_json::to_vec(pair_set)?;
    let digest = Blake2b512::digest(bytes);
    Ok(digest[..PAIR_SET_DIGEST_BYTES]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Remove `path` while undoing an import: `None` when it is gone, or the
/// sentence naming it when it could not be removed and stays behind.
/// Dropped, a committed pair set left unrecorded in pairs/ came back on the
/// next import as a conflict nobody could explain.
fn left_behind(path: &Path) -> Option<String> {
    match fs::remove_file(path) {
        Ok(()) => None,
        Err(error) => Some(format!(
            "{} could not be removed and is left behind: {error}",
            path.display()
        )),
    }
}

fn with_note(failure: anyhow::Error, note: Option<String>) -> anyhow::Error {
    match note {
        Some(note) => failure.context(format!("rolling back the import: {note}")),
        None => failure,
    }
}

/// A workspace name is a file name: it starts with an ASCII letter or digit and
/// holds only letters, digits, dots, underscores and hyphens. How long it may be
/// is the filesystem's to say, and it says so when the set is written.
fn validate_name(name: &str) -> std::result::Result<(), String> {
    let valid = name
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphanumeric)
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err("pair-set name must start with an ASCII letter or digit and contain only letters, digits, dots, underscores, or hyphens".to_owned())
    }
}

/// The id a set imported without --name takes: its file name's letters and
/// digits, lowercased, each run of anything else one hyphen. A name that
/// holds none, or is not UTF-8, gives no id, and the reason names the file
/// and the flag that supplies one.
fn derived_name(source: &Path) -> std::result::Result<String, String> {
    let refused = |why: &str| {
        format!(
            "{} {why}, so no pair-set id can be derived from it; name the set with --name",
            source.display()
        )
    };
    let stem = source
        .file_stem()
        .ok_or_else(|| refused("has no file name"))?
        .to_str()
        .ok_or_else(|| refused("has a file name that is not UTF-8"))?;
    let mut name = String::new();
    let mut separator = false;
    for character in stem.chars() {
        if character.is_ascii_alphanumeric() {
            if separator && !name.is_empty() {
                name.push('-');
            }
            separator = false;
            name.push(character.to_ascii_lowercase());
        } else {
            separator = true;
        }
    }
    if name.is_empty() {
        return Err(refused("has no ASCII letter or digit in its file name"));
    }
    Ok(name)
}
