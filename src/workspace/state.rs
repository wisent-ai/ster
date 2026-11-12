//! Where the workspace lives on disk, what it records, and how a change to
//! it is written so a reader never sees a half-written document.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::failures::{file, refuse, workspace::store};

/// What the workspace document is called in a refusal of its file.
const RECORD: &str = "the Ster workspace";

/// What the file a workspace change is first written to is called.
const TEMPORARY: &str = "the draft of the Ster workspace";

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
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        // Nothing there yet, or a part of the path is a file: the first write
        // creates the directory, or refuses naming the part in the way.
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(WorkspaceState::default());
        }
        Err(error) => return Err(file::read::refusal(RECORD, &path, &error)),
    };
    let state: WorkspaceState = serde_json::from_slice(&bytes)
        .map_err(|error| file::json::refusal(RECORD, store::FORM, &path, &error))?;
    if state.schema != WORKSPACE_SCHEMA {
        return Err(refuse(
            &store::SCHEMA,
            &[
                ("path", &path.display().to_string()),
                ("found", &state.schema),
                ("expected", WORKSPACE_SCHEMA),
            ],
        ));
    }
    Ok(state)
}

pub(super) fn save_state(state: &WorkspaceState) -> Result<()> {
    let root = workspace_root()?;
    create_directory(&root, &root)?;
    let mut bytes = crate::json::pretty(state).into_bytes();
    bytes.push(b'\n');
    atomic_write(&root.join(STATE_FILE), &bytes)
}

/// Create `directory` of the workspace under `root`, or refuse naming it.
pub(super) fn create_directory(root: &Path, directory: &Path) -> Result<()> {
    fs::create_dir_all(directory).map_err(|error| {
        let left = format!("nothing was written under {}", root.display());
        file::write::refusal("creating", "the Ster workspace directory", directory, &left, &error)
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let mut handle = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| {
            let left = format!("nothing was written; {} is unchanged", path.display());
            file::write::refusal("creating", TEMPORARY, &temporary, &left, &error)
        })?;
    if let Err(error) = handle.write_all(bytes).and_then(|_| handle.sync_all()) {
        let left = discard(&temporary);
        return Err(file::write::refusal("writing", TEMPORARY, &temporary, &left, &error));
    }
    drop(handle);
    if let Err(error) = fs::rename(&temporary, path) {
        let left = discard(&temporary);
        let action = format!("renaming {} into place as", temporary.display());
        return Err(file::write::refusal(&action, RECORD, path, &left, &error));
    }
    Ok(())
}

/// What a failed write leaves: its temporary file removed, or that file and
/// why it stayed, for the refusal to name.
fn discard(temporary: &Path) -> String {
    match fs::remove_file(temporary) {
        Ok(()) => format!(
            "{} was removed; the workspace document is unchanged",
            temporary.display()
        ),
        Err(error) => format!(
            "{} was left behind ({}: {error}); the workspace document is unchanged",
            temporary.display(),
            file::kind(&error)
        ),
    }
}

/// The document's file name inside the workspace directory.
const STATE_FILE: &str = "workspace.json";

/// The directory the workspace lives in: `$XDG_DATA_HOME/ster`, else
/// `~/.local/share/ster`.
pub(super) fn workspace_root() -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(root).join("ster"));
    }
    let home = std::env::var_os("HOME").filter(|value| !value.is_empty()).ok_or_else(|| {
        let said = |name: &str| match std::env::var_os(name) {
            None => "unset".to_owned(),
            Some(_) => "set but empty".to_owned(),
        };
        refuse(&store::HOME_UNSET, &[("xdg", &said("XDG_DATA_HOME")), ("home", &said("HOME"))])
    })?;
    Ok(PathBuf::from(home).join(".local/share/ster"))
}

pub(super) fn state_path() -> Result<PathBuf> {
    Ok(workspace_root()?.join(STATE_FILE))
}
