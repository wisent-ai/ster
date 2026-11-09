//! Where the workspace lives on disk, what it records, and how a change to
//! it is written so a reader never sees a half-written document.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::failures::{refuse, workspace::store};

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
    let shown = path.display().to_string();
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
        Err(error) => {
            return Err(refuse(
                &store::UNREADABLE,
                &[("path", &shown), ("error", &error.to_string())],
            ));
        }
    };
    let state: WorkspaceState = serde_json::from_slice(&bytes).map_err(|error| {
        refuse(
            &store::MALFORMED,
            &[("path", &shown), ("error", &error.to_string())],
        )
    })?;
    if state.schema != WORKSPACE_SCHEMA {
        return Err(refuse(
            &store::SCHEMA,
            &[
                ("path", &shown),
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
    let mut bytes = serde_json::to_vec_pretty(state)?;
    bytes.push(b'\n');
    atomic_write(&root, &root.join(STATE_FILE), &bytes)
}

/// Create `directory` of the workspace under `root`, or refuse naming both.
pub(super) fn create_directory(root: &Path, directory: &Path) -> Result<()> {
    fs::create_dir_all(directory).map_err(|error| {
        refuse(
            &store::DIRECTORY,
            &[
                ("directory", &directory.display().to_string()),
                ("root", &root.display().to_string()),
                ("error", &error.to_string()),
            ],
        )
    })
}

fn atomic_write(root: &Path, path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let refused = |step: &str, error: std::io::Error, left: String| {
        refuse(
            &store::WRITE,
            &[
                ("path", &path.display().to_string()),
                ("directory", &root.display().to_string()),
                ("step", step),
                ("error", &error.to_string()),
                ("left", &left),
            ],
        )
    };
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| {
            let step = format!("creating {}", temporary.display());
            refused(&step, error, "nothing was written".to_owned())
        })?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let step = format!("writing {}", temporary.display());
        return Err(refused(&step, error, discard(&temporary)));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, path) {
        let step = format!("replacing {} with {}", path.display(), temporary.display());
        return Err(refused(&step, error, discard(&temporary)));
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
        Err(error) => format!("{} was left behind: {error}", temporary.display()),
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
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| refuse(&store::HOME_UNSET, &[]))?;
    Ok(PathBuf::from(home).join(".local/share/ster"))
}

pub(super) fn state_path() -> Result<PathBuf> {
    Ok(workspace_root()?.join(STATE_FILE))
}
