//! What can be done to a pair set already in the workspace: make it the
//! active input, or take it out. The counterparts of `import_pair_set`, which
//! adds a set and activates it.

use std::fs;

use anyhow::{bail, Context, Result};

use super::state::{load_state, save_state};
use super::{summary, WorkspaceSummary};

fn known(ids: &[String], id: &str) -> Result<()> {
    if ids.iter().any(|known| known == id) {
        return Ok(());
    }
    if ids.is_empty() {
        bail!("the Ster workspace holds no pair set named {id}; it holds none — import one with `ster workspace import-pairs`");
    }
    bail!("the Ster workspace holds no pair set named {id}; it holds {}", ids.join(", "))
}

/// Make `id` the active input every operation reads by default.
pub fn select_pair_set(id: &str) -> Result<WorkspaceSummary> {
    let mut state = load_state()?;
    let ids: Vec<String> = state.pair_sets.iter().map(|entry| entry.id.clone()).collect();
    known(&ids, id)?;
    if state.active_pair_set.as_deref() == Some(id) {
        bail!("pair set {id} is already the active one");
    }
    state.active_pair_set = Some(id.to_owned());
    save_state(&state)?;
    summary()
}

/// Take `id` out of the workspace and delete the workspace's own copy; the
/// source file it was imported from is not touched. Removing the active set
/// leaves no active set, said in the answer, rather than activating another
/// one nobody chose.
pub fn remove_pair_set(id: &str) -> Result<WorkspaceSummary> {
    let mut state = load_state()?;
    let ids: Vec<String> = state.pair_sets.iter().map(|entry| entry.id.clone()).collect();
    known(&ids, id)?;
    let position = state
        .pair_sets
        .iter()
        .position(|entry| entry.id == id)
        .context("pair set vanished between reads")?;
    let entry = state.pair_sets.remove(position);
    if state.active_pair_set.as_deref() == Some(id) {
        state.active_pair_set = None;
    }
    save_state(&state)?;
    match fs::remove_file(&entry.path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!("pair set {id} left the workspace but its copy {} was not deleted", entry.path.display())
            })
        }
    }
    summary()
}
