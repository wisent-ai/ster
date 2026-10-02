//! Where an operator's progress through the walkthrough is kept, and how a
//! run picks it up again.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use super::{JOURNEY_ID, JOURNEY_VERSION, PRODUCT_ID, STATE_SCHEMA, screen_by_id};

pub(super) fn load_or_start_state(definition: &Value) -> Result<Value> {
    let path = state_path();
    if path.exists() {
        let state: Value = serde_json::from_str(
            &fs::read_to_string(&path)
                .with_context(|| format!("read onboarding state {}", path.display()))?,
        )
        .context("parse onboarding state")?;
        if state.get("schema").and_then(Value::as_str) != Some(STATE_SCHEMA)
            || state.get("product_id").and_then(Value::as_str) != Some(PRODUCT_ID)
            || state.get("journey_id").and_then(Value::as_str) != Some(JOURNEY_ID)
            || state.get("journey_version").and_then(Value::as_str) != Some(JOURNEY_VERSION)
        {
            bail!("stored onboarding state identity mismatch; use --reset to replace it");
        }
        let current_screen_id = state
            .get("current_screen_id")
            .and_then(Value::as_str)
            .context("stored onboarding state has no current screen")?;
        screen_by_id(definition, current_screen_id)?;
        return Ok(state);
    }

    let state = fresh_state(definition)?;
    save_state(&state)?;
    Ok(state)
}

pub(super) fn fresh_state(definition: &Value) -> Result<Value> {
    let entry_screen_id = definition
        .get("entry_screen_id")
        .and_then(Value::as_str)
        .context("canonical onboarding journey has no entry screen")?;
    Ok(json!({
        "schema": STATE_SCHEMA,
        "product_id": PRODUCT_ID,
        "journey_id": JOURNEY_ID,
        "journey_version": JOURNEY_VERSION,
        "current_screen_id": entry_screen_id,
        "status": "in_progress",
        "evidence": Map::<String, Value>::new(),
    }))
}

pub(super) fn save_state(state: &Value) -> Result<()> {
    let path = state_path();
    let parent = path
        .parent()
        .context("onboarding state path has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create onboarding state directory {}", parent.display()))?;
    let body = format!("{}\n", serde_json::to_string_pretty(state)?);
    fs::write(&path, body).with_context(|| format!("write onboarding state {}", path.display()))
}

fn state_path() -> PathBuf {
    if let Some(path) = std::env::var_os("XDG_STATE_HOME") {
        return PathBuf::from(path).join("ster/onboarding.json");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".local/state/ster/onboarding.json");
    }
    std::env::temp_dir().join("ster/onboarding.json")
}
