use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use blake2::{Blake2b512, Digest};
use serde::{Deserialize, Serialize};

use crate::PairSet;

const WORKSPACE_SCHEMA: &str = "ster.workspace.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PairSetEntry {
    id: String,
    digest: String,
    path: PathBuf,
    source: PathBuf,
    pair_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceState {
    schema: String,
    #[serde(default)]
    active_pair_set: Option<String>,
    #[serde(default)]
    pair_sets: Vec<PairSetEntry>,
}

impl Default for WorkspaceState {
    fn default() -> Self {
        Self {
            schema: WORKSPACE_SCHEMA.to_owned(),
            active_pair_set: None,
            pair_sets: Vec::new(),
        }
    }
}

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

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceSummary {
    pub schema: &'static str,
    pub active_pair_set: Option<String>,
    pub pair_sets: Vec<WorkspacePairSet>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspacePairSet {
    pub id: String,
    pub path: String,
    pub source: String,
    pub pair_count: usize,
    pub active: bool,
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

pub fn active_pair_set() -> Result<Option<PathBuf>> {
    let state = load_state()?;
    let Some(active) = state.active_pair_set else {
        return Ok(None);
    };
    let entry = state
        .pair_sets
        .iter()
        .find(|entry| entry.id == active)
        .with_context(|| format!("Ster workspace names missing active pair set {active}"))?;
    if !entry.path.is_file() {
        bail!(
            "active Ster pair set is missing: {}; import it again or select another set",
            entry.path.display()
        );
    }
    Ok(Some(entry.path.clone()))
}

pub fn summary() -> Result<WorkspaceSummary> {
    let state = load_state()?;
    let active = state.active_pair_set.clone();
    Ok(WorkspaceSummary {
        schema: WORKSPACE_SCHEMA,
        active_pair_set: active.clone(),
        pair_sets: state
            .pair_sets
            .into_iter()
            .map(|entry| WorkspacePairSet {
                active: active.as_deref() == Some(entry.id.as_str()),
                id: entry.id,
                path: entry.path.display().to_string(),
                source: entry.source.display().to_string(),
                pair_count: entry.pair_count,
            })
            .collect(),
    })
}

fn pair_set_digest(pair_set: &PairSet) -> Result<String> {
    let bytes = serde_json::to_vec(pair_set)?;
    let digest = Blake2b512::digest(bytes);
    Ok(digest[..16].iter().map(|byte| format!("{byte:02x}")).collect())
}

fn validate_name(name: &str) -> std::result::Result<(), String> {
    let valid = !name.is_empty()
        && name.len() <= 64
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
        if name.len() == 64 {
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

fn load_state() -> Result<WorkspaceState> {
    let path = state_path()?;
    if !path.exists() {
        return Ok(WorkspaceState::default());
    }
    let bytes = fs::read(&path)
        .with_context(|| format!("failed to read Ster workspace {}", path.display()))?;
    let state: WorkspaceState = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid Ster workspace JSON in {}", path.display()))?;
    if state.schema != WORKSPACE_SCHEMA {
        bail!("unsupported Ster workspace schema in {}", path.display());
    }
    Ok(state)
}

fn save_state(state: &WorkspaceState) -> Result<()> {
    let path = state_path()?;
    let parent = path.parent().context("Ster workspace path has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create Ster workspace directory {}", parent.display()))?;
    let mut bytes = serde_json::to_vec_pretty(state)?;
    bytes.push(b'\n');
    atomic_write(&path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .with_context(|| format!("failed to create {}", temporary.display()))?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(error).with_context(|| format!("failed to write {}", temporary.display()));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error).with_context(|| format!("failed to replace {}", path.display()));
    }
    Ok(())
}

fn workspace_root() -> Result<PathBuf> {
    Ok(state_path()?
        .parent()
        .context("Ster workspace path has no parent")?
        .to_path_buf())
}

fn state_path() -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(root).join("ster/workspace.json"));
    }
    let home = std::env::var_os("HOME").filter(|value| !value.is_empty()).context(
        "HOME is unavailable; set HOME or XDG_DATA_HOME before importing a Ster pair set",
    )?;
    Ok(PathBuf::from(home).join(".local/share/ster/workspace.json"))
}
