//! Where an operator's progress through the walkthrough is kept, and how a
//! run picks it up again.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use ster::failures::{file, onboarding as causes, refuse};

use super::journey::Journey;
use super::{JOURNEY_ID, JOURNEY_VERSION, PRODUCT_ID, STATE_SCHEMA};

/// What the onboarding state is called in a refusal of its file.
const ROLE: &str = "the onboarding state";

/// The progress one account made through the journey.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct State {
    schema: String,
    product_id: String,
    journey_id: String,
    journey_version: String,
    pub(super) current_screen_id: String,
    pub(super) status: String,
    #[serde(default)]
    pub(super) evidence: Map<String, Value>,
}

/// The state this account keeps, or a fresh one written at the journey's
/// entry when it keeps none.
pub(super) fn load_or_start_state(journey: &Journey) -> Result<State> {
    let path = state_path()?;
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let state = fresh_state(journey);
            save_state(&state)?;
            return Ok(state);
        }
        Err(error) => return Err(file::read::refusal(ROLE, &path, &error)),
    };
    let state: State = serde_json::from_slice(&bytes)
        .map_err(|error| file::json::refusal(ROLE, causes::FORM, &path, &error))?;
    for (field, found, expected) in [
        ("schema", &state.schema, STATE_SCHEMA),
        ("product_id", &state.product_id, PRODUCT_ID),
        ("journey_id", &state.journey_id, JOURNEY_ID),
        ("journey_version", &state.journey_version, JOURNEY_VERSION),
    ] {
        if found != expected {
            return Err(refuse(
                &causes::STATE_FOREIGN,
                &[
                    ("path", &path.display().to_string()),
                    ("field", field),
                    ("found", &format!("{found:?}")),
                    ("expected", &format!("{expected:?}")),
                ],
            ));
        }
    }
    known(journey, &state, &path)?;
    Ok(state)
}

/// Refuses a state standing at a screen the journey does not hold.
fn known(journey: &Journey, state: &State, path: &Path) -> Result<()> {
    if journey.screen(&state.current_screen_id).is_some() {
        return Ok(());
    }
    Err(refuse(
        &causes::SCREEN_UNKNOWN,
        &[
            ("path", &path.display().to_string()),
            ("screen", &format!("{:?}", state.current_screen_id)),
            ("screens", &journey.screen_ids()),
        ],
    ))
}

/// The state of a walkthrough that has not begun: at the journey's entry.
pub(super) fn fresh_state(journey: &Journey) -> State {
    State {
        schema: STATE_SCHEMA.to_owned(),
        product_id: PRODUCT_ID.to_owned(),
        journey_id: JOURNEY_ID.to_owned(),
        journey_version: JOURNEY_VERSION.to_owned(),
        current_screen_id: journey.entry_screen_id.clone(),
        status: "in_progress".to_owned(),
        evidence: Map::new(),
    }
}

/// Writes `state` where this account keeps it.
pub(super) fn save_state(state: &State) -> Result<()> {
    let path = state_path()?;
    let directory = file::parent(&path);
    let left = "the progress shown is not kept, and the state that was there is unchanged";
    fs::create_dir_all(&directory).map_err(|error| {
        file::write::refusal("creating the directory of", ROLE, &path, left, &error)
    })?;
    let body = format!("{}\n", ster::json::pretty(state));
    fs::write(&path, body).map_err(|error| {
        let left =
            "the progress shown is not kept, and the state there may be emptied or cut short";
        file::write::refusal("writing", ROLE, &path, left, &error)
    })
}

/// `$XDG_STATE_HOME/ster`, else `$HOME/.local/state/ster`. With neither it
/// is refused: the temporary directory it used instead is cleared by the
/// system and shared by every account, so progress vanished or crossed users.
fn state_path() -> Result<PathBuf> {
    let xdg = std::env::var_os("XDG_STATE_HOME").filter(|value| !value.is_empty());
    if let Some(path) = xdg {
        return Ok(PathBuf::from(path).join("ster/onboarding.json"));
    }
    match std::env::var_os("HOME").filter(|value| !value.is_empty()) {
        Some(home) => Ok(PathBuf::from(home).join(".local/state/ster/onboarding.json")),
        None => Err(refuse(
            &causes::HOME_UNSET,
            &[("xdg", "unset or empty"), ("home", "unset or empty")],
        )),
    }
}
