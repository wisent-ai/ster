use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

const PRODUCT_ID: &str = "ster";
const JOURNEY_ID: &str = "first-use";
const JOURNEY_VERSION: &str = "2026-09-05.1";
const FIRST_SUCCESS_FACT: &str = "contrastive_pair_set_imported";
const STATE_SCHEMA: &str = "ster.onboarding-state.v1";
const DEFINITION: &str = include_str!("onboarding_first_use.json");

pub fn run(reset: bool, import_pairs: Option<&Path>, name: Option<&str>) -> Result<()> {
    let definition = canonical_definition()?;
    let mut state = if reset {
        let state = fresh_state(&definition)?;
        save_state(&state)?;
        println!(
            "Ster first-use journey reset: recorded progress and evidence discarded. Showing the walkthrough again now."
        );
        state
    } else {
        load_or_start_state(&definition)?
    };

    if let Some(source) = import_pairs {
        let report = ster::workspace::import_pair_set(source, name)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        if report.accepted() {
            let destination = report
                .path
                .as_deref()
                .context("accepted pair-set import did not return a destination")?;
            record_pair_set_imported(Path::new(destination))?;
            println!(
                "Ster first-use journey complete: the imported pair set is active for training and evaluation."
            );
        } else {
            bail!(
                "{}",
                report
                    .reason
                    .as_deref()
                    .unwrap_or("Ster did not accept the pair set")
            );
        }
        return Ok(());
    }

    if state.get("status").and_then(Value::as_str) == Some("completed") {
        println!(
            "Ster first-use journey is already complete: a contrastive pair set was imported."
        );
        return Ok(());
    }

    loop {
        let screen_id = state
            .get("current_screen_id")
            .and_then(Value::as_str)
            .context("stored onboarding state has no current screen")?
            .to_string();
        let screen = screen_by_id(&definition, &screen_id)?;
        render(screen)?;

        let Some(next_screen_id) = next_screen_id(screen)? else {
            save_state(&state)?;
            println!("\nFirst-use progress saved; import an existing pair set with the command shown above.");
            return Ok(());
        };

        state
            .as_object_mut()
            .context("onboarding state is not an object")?
            .insert(
                "current_screen_id".to_string(),
                Value::String(next_screen_id),
            );
        save_state(&state)?;
    }
}

/// Record onboarding evidence only after the workspace accepted and persisted
/// an imported pair set. Merely choosing a file never completes first use.
pub fn record_pair_set_imported(path: &Path) -> Result<()> {
    let definition = canonical_definition()?;
    let mut state = load_or_start_state(&definition)?;
    let terminal_screen_id = definition
        .get("screens")
        .and_then(Value::as_array)
        .and_then(|screens| {
            screens.iter().find(|screen| {
                screen
                    .get("completion_evidence")
                    .and_then(|evidence| evidence.get("fact"))
                    .and_then(Value::as_str)
                    == Some(FIRST_SUCCESS_FACT)
            })
        })
        .and_then(|screen| screen.get("screen_id"))
        .and_then(Value::as_str)
        .context("canonical onboarding journey has no first-success screen")?;

    let object = state
        .as_object_mut()
        .context("onboarding state is not an object")?;
    object.insert(
        "current_screen_id".to_string(),
        Value::String(terminal_screen_id.to_string()),
    );
    object.insert("status".to_string(), Value::String("completed".to_string()));
    object.insert(
        "evidence".to_string(),
        json!({
            FIRST_SUCCESS_FACT: true,
            "pair_set_path": path.display().to_string(),
        }),
    );
    save_state(&state)
}

fn canonical_definition() -> Result<Value> {
    let definition: Value =
        serde_json::from_str(DEFINITION).context("parse canonical onboarding journey")?;
    if definition.get("schema_version").and_then(Value::as_u64) != Some(1)
        || definition.get("product_id").and_then(Value::as_str) != Some(PRODUCT_ID)
        || definition.get("journey_id").and_then(Value::as_str) != Some(JOURNEY_ID)
        || definition.get("journey_version").and_then(Value::as_str) != Some(JOURNEY_VERSION)
        || definition.get("first_success_fact").and_then(Value::as_str)
            != Some(FIRST_SUCCESS_FACT)
    {
        bail!("canonical onboarding journey identity mismatch");
    }

    let entry_screen_id = definition
        .get("entry_screen_id")
        .and_then(Value::as_str)
        .context("canonical onboarding journey has no entry screen")?;
    let screens = definition
        .get("screens")
        .and_then(Value::as_array)
        .context("canonical onboarding journey has no screens")?;
    if !(3..=5).contains(&screens.len()) {
        bail!("canonical onboarding journey must have three to five screens");
    }

    let mut screen_ids = Vec::with_capacity(screens.len());
    for screen in screens {
        let screen_id = screen
            .get("screen_id")
            .and_then(Value::as_str)
            .context("canonical onboarding screen has no id")?;
        if screen_ids.contains(&screen_id) {
            bail!("duplicate canonical onboarding screen id: {screen_id}");
        }
        screen_ids.push(screen_id);
        let presentation = screen
            .get("presentation")
            .and_then(Value::as_object)
            .context("canonical onboarding screen has no presentation")?;
        presentation
            .get("title")
            .and_then(Value::as_str)
            .context("canonical onboarding screen has no title")?;
        presentation
            .get("body")
            .and_then(Value::as_str)
            .context("canonical onboarding screen has no body")?;
    }
    if !screen_ids.contains(&entry_screen_id) {
        bail!("canonical onboarding entry screen does not exist");
    }

    for screen in screens {
        for transition in screen
            .get("transitions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let target = transition
                .get("next_screen_id")
                .and_then(Value::as_str)
                .context("canonical onboarding transition has no target")?;
            if !screen_ids.contains(&target) {
                bail!("canonical onboarding transition target does not exist: {target}");
            }
        }
    }
    Ok(definition)
}

fn screen_by_id<'a>(definition: &'a Value, screen_id: &str) -> Result<&'a Value> {
    definition
        .get("screens")
        .and_then(Value::as_array)
        .and_then(|screens| {
            screens
                .iter()
                .find(|screen| screen.get("screen_id").and_then(Value::as_str) == Some(screen_id))
        })
        .with_context(|| format!("canonical onboarding screen is unavailable: {screen_id}"))
}

fn next_screen_id(screen: &Value) -> Result<Option<String>> {
    let Some(transitions) = screen.get("transitions").and_then(Value::as_array) else {
        return Ok(None);
    };
    transitions
        .iter()
        .max_by_key(|transition| {
            transition
                .get("priority")
                .and_then(Value::as_i64)
                .unwrap_or_default()
        })
        .map(|transition| {
            transition
                .get("next_screen_id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .context("canonical onboarding transition has no target")
        })
        .transpose()
}

fn render(screen: &Value) -> Result<()> {
    let presentation = screen
        .get("presentation")
        .and_then(Value::as_object)
        .context("canonical onboarding screen has no presentation")?;
    let title = presentation
        .get("title")
        .and_then(Value::as_str)
        .context("canonical onboarding screen has no title")?;
    let body = presentation
        .get("body")
        .and_then(Value::as_str)
        .context("canonical onboarding screen has no body")?;
    println!("\n{title}\n{body}");
    Ok(())
}

fn load_or_start_state(definition: &Value) -> Result<Value> {
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

fn fresh_state(definition: &Value) -> Result<Value> {
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

fn save_state(state: &Value) -> Result<()> {
    let path = state_path();
    let parent = path
        .parent()
        .context("onboarding state path has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create onboarding state directory {}", parent.display()))?;
    let body = format!("{}\n", serde_json::to_string_pretty(state)?);
    fs::write(&path, body)
        .with_context(|| format!("write onboarding state {}", path.display()))
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
