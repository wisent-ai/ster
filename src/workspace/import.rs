//! Taking a pair set someone else produced into the workspace: what is
//! accepted, what the copy is named, and when a repeat is the same import
//! rather than a new one.

use std::{fs, path::Path};

use anyhow::{bail, Context, Result};
use blake2::{Blake2b512, Digest};
use serde::Serialize;

use crate::PairSet;

use super::state::{load_state, save_state, workspace_root, PairSetEntry};

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub status: &'static str,
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
    pub fn accepted(&self) -> bool {
        matches!(self.status, "imported" | "unchanged")
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
                status: "rejected",
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
    let id = match requested_name {
        Some(name) => match validate_name(name) {
            Ok(()) => name.to_owned(),
            Err(reason) => {
                return Ok(ImportReport {
                    status: "rejected",
                    source: source.display().to_string(),
                    id: None,
                    path: None,
                    pair_count: Some(pair_set.pairs.len()),
                    reason: Some(reason),
                });
            }
        },
        None => derived_name(&source),
    };
    let digest = pair_set_digest(&pair_set)?;
    let mut state = load_state()?;

    if let Some(existing) = state.pair_sets.iter().find(|entry| entry.digest == digest).cloned() {
        state.active_pair_set = Some(existing.id.clone());
        save_state(&state)?;
        return Ok(ImportReport {
            status: "unchanged",
            source: source.display().to_string(),
            id: Some(existing.id),
            path: Some(existing.path.display().to_string()),
            pair_count: Some(existing.pair_count),
            reason: None,
        });
    }

    if let Some(existing) = state.pair_sets.iter().find(|entry| entry.id == id) {
        return Ok(ImportReport {
            status: "conflicting",
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
    fs::create_dir_all(&pairs_dir)
        .with_context(|| format!("failed to create Ster pair-set directory {}", pairs_dir.display()))?;
    let destination = pairs_dir.join(format!("{id}.json"));
    if destination.exists() {
        return Ok(ImportReport {
            status: "conflicting",
            source: source.display().to_string(),
            id: Some(id),
            path: Some(destination.display().to_string()),
            pair_count: Some(pair_set.pairs.len()),
            reason: Some("the destination already exists outside the Ster workspace index; it was not replaced".to_owned()),
        });
    }

    let temporary = pairs_dir.join(format!(".{id}.{}.incoming", std::process::id()));
    if temporary.exists() {
        bail!("Ster import staging path already exists: {}", temporary.display());
    }
    pair_set.save(&temporary)?;
    if let Err(error) = fs::hard_link(&temporary, &destination) {
        let _ = fs::remove_file(&temporary);
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            return Ok(ImportReport {
                status: "conflicting",
                source: source.display().to_string(),
                id: Some(id),
                path: Some(destination.display().to_string()),
                pair_count: Some(pair_set.pairs.len()),
                reason: Some("the destination appeared while the import was being committed; it was not replaced".to_owned()),
            });
        }
        return Err(error).with_context(|| {
            format!(
                "failed to commit imported pair set {}",
                destination.display()
            )
        });
    }
    if let Err(error) = fs::remove_file(&temporary) {
        let _ = fs::remove_file(&destination);
        return Err(error)
            .with_context(|| format!("failed to remove import staging file {}", temporary.display()));
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
        let _ = fs::remove_file(&destination);
        return Err(error);
    }

    Ok(ImportReport {
        status: "imported",
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
    Ok(digest[..16].iter().map(|byte| format!("{byte:02x}")).collect())
}

/// A workspace name is at most this many bytes, so it fits a file name on every filesystem.
const MAX_NAME_LENGTH: usize = 64;

fn validate_name(name: &str) -> std::result::Result<(), String> {
    let valid = !name.is_empty()
        && name.len() <= MAX_NAME_LENGTH
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        && name.as_bytes()[0].is_ascii_alphanumeric();
    if valid {
        Ok(())
    } else {
        Err("pair-set name must start with an ASCII letter or digit and contain at most 64 letters, digits, dots, underscores, or hyphens".to_owned())
    }
}

fn derived_name(source: &Path) -> String {
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("pairs");
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
        if name.len() == MAX_NAME_LENGTH {
            break;
        }
    }
    let name = name.trim_end_matches('-');
    if name.is_empty() {
        "pairs".to_owned()
    } else {
        name.to_owned()
    }
}
