//! Where the workspace lives on disk, what it records, and how a change to
//! it is written so a reader never sees a half-written document.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub(super) const WORKSPACE_SCHEMA: &str = "ster.workspace.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PairSetEntry {
    pub(super) id: String,
    pub(super) digest: String,
    pub(super) path: PathBuf,
    pub(super) source: PathBuf,
    pub(super) pair_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkspaceState {
    schema: String,
    #[serde(default)]
    pub(super) active_pair_set: Option<String>,
    #[serde(default)]
    pub(super) pair_sets: Vec<PairSetEntry>,
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

pub(super) fn load_state() -> Result<WorkspaceState> {
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

pub(super) fn save_state(state: &WorkspaceState) -> Result<()> {
    let path = state_path()?;
    let parent = path.parent().context("Ster workspace path has no parent")?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create Ster workspace directory {}",
            parent.display()
        )
    })?;
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

pub(super) fn workspace_root() -> Result<PathBuf> {
    Ok(state_path()?
        .parent()
        .context("Ster workspace path has no parent")?
        .to_path_buf())
}

fn state_path() -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(root).join("ster/workspace.json"));
    }
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .context(
            "HOME is unavailable; set HOME or XDG_DATA_HOME before importing a Ster pair set",
        )?;
    Ok(PathBuf::from(home).join(".local/share/ster/workspace.json"))
}
