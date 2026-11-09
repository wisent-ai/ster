//! Ster's persistent local workspace: the pair sets a person imported, which
//! one is active, and where the copies live.
//!
//! `state` owns the document on disk and how it is written; `import` owns
//! taking a new set in. What is left here is what a reader asks the
//! workspace: which set is active, and what does it hold.

use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;

use crate::failures::{refuse, workspace::manage as causes};

mod import;
mod manage;
mod name;
mod state;

pub use import::{ImportReport, import_pair_set};
pub use manage::{remove_pair_set, select_pair_set};

use state::{WORKSPACE_SCHEMA, WorkspaceState, load_state};

/// The ids the workspace holds, as a refusal lists them, or `none`.
fn held(state: &WorkspaceState) -> String {
    if state.pair_sets.is_empty() {
        return "none".to_owned();
    }
    state
        .pair_sets
        .iter()
        .map(|entry| entry.id.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The copy of the active imported set, which a command given no --pairs
/// reads; refused when no set is active, when the workspace names an active
/// set it does not hold, or when that set's copy is gone.
pub fn active_pair_set() -> Result<PathBuf> {
    let state = load_state()?;
    let Some(active) = state.active_pair_set.as_deref() else {
        return Err(refuse(&causes::NO_ACTIVE, &[("held", &held(&state))]));
    };
    let Some(entry) = state.pair_sets.iter().find(|entry| entry.id == active) else {
        return Err(refuse(
            &causes::ACTIVE_UNKNOWN,
            &[("id", active), ("held", &held(&state))],
        ));
    };
    if !entry.path.is_file() {
        return Err(refuse(
            &causes::ACTIVE_COPY_MISSING,
            &[("id", active), ("path", &entry.path.display().to_string())],
        ));
    }
    Ok(entry.path.clone())
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
