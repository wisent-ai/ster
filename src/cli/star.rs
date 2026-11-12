//! The GitHub star request: after a person's fifth successful run of a
//! model (`vector train`, `vector optimize`, `generate`, `decide`, or a
//! `tune` objective that fits an adapter, from the CLI or from Ster Desktop
//! through `ster request`) Ster asks once for a star on its public
//! repository. The request is the `star-request` journey every Wisent
//! product walks, run by the shared onboarding runtime and kept per user in
//! `$XDG_STATE_HOME/ster/onboarding.json` (default
//! `~/.local/state/ster/onboarding.json`); `ster star` and the request
//! operations `star/status|dismiss|reset` read, dismiss and reset it. The
//! threshold is the runtime's, the operator's number; the repository is this
//! package's `repository`, the catalog's `star_repository`.

use std::path::Path;

use anyhow::Result;
use clap::Subcommand;
use serde_json::Value;
use ster::failures::{file, refuse, star as causes};
use wisent_onboarding_client::{
    JourneyError, ProductStarRequest, StarProduct, StarRequest, state_file,
};

#[derive(Debug, Subcommand)]
pub(super) enum StarCommand {
    /// Where the request stands: counting, shown or dismissed.
    Status,
    /// Never make the request again, until a reset.
    Dismiss,
    /// Count again from zero; set aside a journey file that holds no state.
    Reset,
}

/// The repository the request asks a star for, the catalog's
/// `star_repository`.
const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

fn request() -> Result<ProductStarRequest, JourneyError> {
    let product = StarProduct::from_package("ster", "Ster", REPOSITORY)?;
    ProductStarRequest::new(product)
}

/// The declared refusal of `step` (`counting a run toward`, `reading`) when
/// the onboarding runtime answered `error`: its kind decides the cause, its
/// sentence is quoted beside the state file and what decides where it is.
pub(crate) fn refusal(step: &str, error: &JourneyError) -> anyhow::Error {
    let cause = match error {
        JourneyError::Storage(_) => &causes::STORAGE,
        JourneyError::Invalid(_) => &causes::INVALID,
        JourneyError::Transport(_) => &causes::TRANSPORT,
        JourneyError::NotStarted => &causes::NOT_STARTED,
    };
    let variable = |name: &str| {
        std::env::var_os(name).map_or_else(|| "unset".to_owned(), |value| value.to_string_lossy().into_owned())
    };
    let (shown, state, directory, directory_state) = match state_file("ster") {
        Ok(path) => {
            let directory = path.parent().unwrap_or(Path::new("/")).to_path_buf();
            (
                path.display().to_string(),
                file::state(&path),
                directory.display().to_string(),
                file::state(&directory),
            )
        }
        Err(_) => {
            let none = "no path (neither XDG_STATE_HOME nor HOME is set)".to_owned();
            (none.clone(), "unknown".to_owned(), none, "unknown".to_owned())
        }
    };
    refuse(
        cause,
        &[
            ("step", step),
            ("file", &shown),
            ("state", &state),
            ("directory", &directory),
            ("directory_state", &directory_state),
            ("xdg", &variable("XDG_STATE_HOME")),
            ("home", &variable("HOME")),
            ("user", &variable("USER")),
            ("repository", REPOSITORY),
            ("said", &error.to_string()),
        ],
    )
}

/// Count one run that completed: the request when this run makes it.
pub(crate) fn observed() -> Result<Option<StarRequest>> {
    request()
        .and_then(|request| request.observe())
        .map_err(|error| refusal("counting a run toward", &error))
}

/// Count one CLI run that completed, when a person reads this terminal. The
/// request, on the run that makes it, goes to standard error, so standard
/// output keeps only the answer; a request that cannot be kept is said there
/// too, by its declared cause, and the run's own answer stands.
pub(super) fn observe() {
    match request() {
        Ok(request) => {
            request.observe_reporting();
        }
        Err(error) => eprintln!(
            "ster: this run is not counted toward the GitHub star request: {}",
            refusal("counting a run toward", &error)
        ),
    }
}

/// The status after `verb` (`status`, `dismiss` or `reset`), as the JSON
/// document `ster star` prints.
pub(crate) fn status_document(verb: &str) -> Result<Value> {
    let step = match verb {
        "dismiss" => "dismissing",
        "reset" => "resetting",
        _ => "reading",
    };
    let request = request().map_err(|error| refusal(step, &error))?;
    let status = match verb {
        "dismiss" => request.dismiss(),
        "reset" => request.reset(),
        _ => request.status(),
    }
    .map_err(|error| refusal(step, &error))?;
    Ok(ster::json::value(&status))
}

/// `ster star status|dismiss|reset`: the status as JSON, or `path: value`
/// lines with `--text`.
pub(super) fn run(command: StarCommand) -> Result<()> {
    let verb = match command {
        StarCommand::Status => "status",
        StarCommand::Dismiss => "dismiss",
        StarCommand::Reset => "reset",
    };
    super::answer_document(&status_document(verb)?)
}
