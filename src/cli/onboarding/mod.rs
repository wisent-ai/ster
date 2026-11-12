//! `ster onboarding`: the first-use walkthrough, shown from the screen the
//! operator reached, and completed once a pair set is imported.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

const PRODUCT_ID: &str = "ster";
const JOURNEY_ID: &str = "first-use";
const JOURNEY_VERSION: &str = "2026-09-05.1";
const FIRST_SUCCESS_FACT: &str = "contrastive_pair_set_imported";
const STATE_SCHEMA: &str = "ster.onboarding-state.v1";

mod journey;
mod state;

use journey::{Screen, journey};
use state::{fresh_state, load_or_start_state, save_state};
pub(super) use state::state_path;

pub fn run(reset: bool, import_pairs: Option<&Path>, name: Option<&str>) -> Result<()> {
    let journey = journey();
    let mut state = if reset {
        let state = fresh_state(&journey);
        save_state(&state)?;
        println!(
            "Ster first-use journey reset: recorded progress and evidence discarded. Showing the walkthrough again now."
        );
        state
    } else {
        load_or_start_state(&journey)?
    };

    if let Some(source) = import_pairs {
        let mut report = ster::workspace::import_pair_set(source, name)?;
        let refusal = report.take_refusal();
        super::answer(&report)?;
        if let Some(refusal) = refusal {
            return Err(refusal);
        }
        // An import that was not refused is imported or unchanged, and both
        // name the copy the workspace holds.
        let destination = report
            .path
            .as_deref()
            .expect("an accepted import names the workspace copy");
        record_pair_set_imported(Path::new(destination))?;
        println!(
            "Ster first-use journey complete: the imported pair set is active for training and evaluation."
        );
        return Ok(());
    }

    if state.status == "completed" {
        println!(
            "Ster first-use journey is already complete: a contrastive pair set was imported."
        );
        return Ok(());
    }

    loop {
        // `load_or_start_state` refused a screen the journey does not hold,
        // and every transition leads to one (`journey` checks it).
        let screen = journey
            .screen(&state.current_screen_id)
            .expect("the state stands at a screen of the journey");
        render(screen);
        let Some(next) = screen.next() else {
            save_state(&state)?;
            println!(
                "\nFirst-use progress saved; import an existing pair set with the command shown above."
            );
            return Ok(());
        };
        state.current_screen_id = next.to_owned();
        save_state(&state)?;
    }
}

/// Record onboarding evidence only after the workspace accepted and persisted
/// an imported pair set. Merely choosing a file never completes first use.
pub fn record_pair_set_imported(path: &Path) -> Result<()> {
    let journey = journey();
    let mut state = load_or_start_state(&journey)?;
    // `journey` checks that one screen is the first success.
    let terminal = journey
        .first_success()
        .expect("the journey names its first-success screen");
    state.current_screen_id = terminal.screen_id.clone();
    state.status = "completed".to_owned();
    state.evidence = [
        (FIRST_SUCCESS_FACT.to_owned(), Value::Bool(true)),
        (
            "pair_set_path".to_owned(),
            Value::String(path.display().to_string()),
        ),
    ]
    .into_iter()
    .collect();
    save_state(&state)
}

fn render(screen: &Screen) {
    println!(
        "\n{}\n{}",
        screen.presentation.title, screen.presentation.body
    );
}
