//! Ster's persistent local workspace: the pair sets a person imported, which
//! one is active, and where the copies live.
//!
//! `state` owns the document on disk and how it is written; `import` owns
//! taking a new set in. What is left here is what a reader asks the
//! workspace: which set is active, and what does it hold.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::Serialize;

mod import;
mod manage;
mod state;

pub use import::{import_pair_set, ImportReport};
pub use manage::{remove_pair_set, select_pair_set};

use state::{load_state, WORKSPACE_SCHEMA};

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

