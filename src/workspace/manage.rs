//! What can be done to a pair set already in the workspace: make it the
//! active input, or take it out. The counterparts of `import_pair_set`, which
//! adds a set and activates it.

use std::fs;

use anyhow::Result;

use crate::failures::{
    refuse,
    workspace::{manage as causes, store},
};

use super::state::{WorkspaceState, load_state, save_state, state_path, workspace_root};
use super::{WorkspaceSummary, held, summary};

/// Where in the workspace's list `id` is, or the refusal naming the ids it
/// holds.
fn position(state: &WorkspaceState, id: &str) -> Result<usize> {
    if let Some(position) = state.pair_sets.iter().position(|entry| entry.id == id) {
        return Ok(position);
    }
    let held = match held(state).as_str() {
        "none" => "none — import one with `ster workspace import`".to_owned(),
        ids => ids.to_owned(),
    };
    let path = state_path()?.display().to_string();
    Err(refuse(
        &store::SET_UNKNOWN,
        &[("id", id), ("held", &held), ("path", &path)],
    ))
}

/// Make `id` the active input every operation reads by default.
pub fn select_pair_set(id: &str) -> Result<WorkspaceSummary> {
    let mut state = load_state()?;
    position(&state, id)?;
    if state.active_pair_set.as_deref() == Some(id) {
        return Err(refuse(&causes::ALREADY_ACTIVE, &[("id", id)]));
    }
    state.active_pair_set = Some(id.to_owned());
    save_state(&state)?;
    summary()
}

/// Take `id` out of the workspace and delete the workspace's own copy; the
/// source file it was imported from is not touched. Removing the active set
/// leaves no active set, said in the answer, rather than activating another
/// one nobody chose.
///
/// The copy goes first: a copy that cannot be deleted leaves the workspace
/// as it was, so the same command removes it once the directory allows it.
/// Taking the entry out first would leave a file no entry records, which the
/// next import of that id meets as a conflict nothing can remove.
pub fn remove_pair_set(id: &str) -> Result<WorkspaceSummary> {
    let mut state = load_state()?;
    let index = position(&state, id)?;
    let entry = state.pair_sets.remove(index);
    match fs::remove_file(&entry.path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            let directory = entry
                .path
                .parent()
                .map_or_else(workspace_root, |parent| Ok(parent.to_path_buf()))?;
            return Err(refuse(
                &causes::COPY_UNDELETABLE,
                &[
                    ("id", id),
                    ("path", &entry.path.display().to_string()),
                    ("directory", &directory.display().to_string()),
                    ("error", &error.to_string()),
                ],
            ));
        }
    }
    if state.active_pair_set.as_deref() == Some(id) {
        state.active_pair_set = None;
    }
    save_state(&state)?;
    summary()
}
