use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

const PRODUCT_ID: &str = "ster";
const JOURNEY_ID: &str = "first-use";
const JOURNEY_VERSION: &str = "2026-09-05.1";
const FIRST_SUCCESS_FACT: &str = "contrastive_pair_set_imported";
const STATE_SCHEMA: &str = "ster.onboarding-state.v1";
const DEFINITION: &str = include_str!("first_use.json");

mod state;

use state::{fresh_state, load_or_start_state, save_state};

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
        let mut report = ster::workspace::import_pair_set(source, name)?;
        let refusal = report.take_refusal();
        super::answer(&report)?;
        if let Some(refusal) = refusal {
            return Err(refusal);
        }
        let destination = report
            .path
            .as_deref()
            .context("accepted pair-set import did not return a destination")?;
        record_pair_set_imported(Path::new(destination))?;
        println!(
            "Ster first-use journey complete: the imported pair set is active for training and evaluation."
        );
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
            println!(
                "\nFirst-use progress saved; import an existing pair set with the command shown above."
            );
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

/// The journey compiled into this binary, checked against what the reader
/// relies on. It is embedded at build time, so it is always the file this code
/// was written with: who it is for, its screens, and every transition's
/// target and priority.
fn canonical_definition() -> Result<Value> {
    let definition: Value =
        serde_json::from_str(DEFINITION).context("parse canonical onboarding journey")?;
    if definition.get("product_id").and_then(Value::as_str) != Some(PRODUCT_ID)
        || definition.get("journey_id").and_then(Value::as_str) != Some(JOURNEY_ID)
        || definition.get("journey_version").and_then(Value::as_str) != Some(JOURNEY_VERSION)
        || definition.get("first_success_fact").and_then(Value::as_str) != Some(FIRST_SUCCESS_FACT)
    {
        bail!(
            "canonical onboarding journey identity mismatch: it names journey {} version {}, this build knows {JOURNEY_ID} version {JOURNEY_VERSION}",
            definition
                .get("journey_id")
                .and_then(Value::as_str)
                .unwrap_or("none"),
            definition
                .get("journey_version")
                .and_then(Value::as_str)
                .unwrap_or("none")
        );
    }

    let entry_screen_id = definition
        .get("entry_screen_id")
        .and_then(Value::as_str)
        .context("canonical onboarding journey has no entry screen")?;
    let screens = definition
        .get("screens")
        .and_then(Value::as_array)
        .context("canonical onboarding journey has no screens")?;
    if screens.is_empty() {
        bail!("canonical onboarding journey {JOURNEY_ID} has an empty screens list");
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
        bail!(
            "canonical onboarding entry screen {entry_screen_id} does not exist among {screen_ids:?}"
        );
    }

    for screen in screens {
        let screen_id = screen
            .get("screen_id")
            .and_then(Value::as_str)
            .context("canonical onboarding screen has no id")?;
        let mut priorities = std::collections::HashSet::new();
        for transition in screen
            .get("transitions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let (target, priority) = edge(screen_id, transition)?;
            if !screen_ids.contains(&target) {
                bail!("canonical onboarding transition target does not exist: {target}");
            }
            if !priorities.insert(priority) {
                bail!(
                    "canonical onboarding transitions of {screen_id} share priority {priority}; the journey contract requires them unique"
                );
            }
        }
    }
    Ok(definition)
}

pub(super) fn screen_by_id<'a>(definition: &'a Value, screen_id: &str) -> Result<&'a Value> {
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

/// One transition's target and priority, refused naming the screen and the
/// target when either is missing or the priority is not a non-negative whole
/// number, the journey contract's type for it.
fn edge<'a>(screen_id: &str, transition: &'a Value) -> Result<(&'a str, u64)> {
    let target = transition
        .get("next_screen_id")
        .and_then(Value::as_str)
        .with_context(|| format!("canonical onboarding transition of {screen_id} has no target"))?;
    let priority = transition
        .get("priority")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!(
                "canonical onboarding transition from {screen_id} to {target} declares no non-negative whole-number priority"
            )
        })?;
    Ok((target, priority))
}

/// The target of the transition that leads: the lowest priority, as the
/// journey contract and Echo's own routing order transitions. Priorities are
/// unique per screen, which the definition check enforces.
fn next_screen_id(screen: &Value) -> Result<Option<String>> {
    let Some(transitions) = screen.get("transitions").and_then(Value::as_array) else {
        return Ok(None);
    };
    let screen_id = screen
        .get("screen_id")
        .and_then(Value::as_str)
        .context("canonical onboarding screen has no id")?;
    let mut chosen: Option<(u64, &str)> = None;
    for transition in transitions {
        let (target, rank) = edge(screen_id, transition)?;
        if chosen.is_none_or(|(best, _)| rank < best) {
            chosen = Some((rank, target));
        }
    }
    Ok(chosen.map(|(_, target)| target.to_string()))
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
