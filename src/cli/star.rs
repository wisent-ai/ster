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

use anyhow::{Result, anyhow};
use clap::Subcommand;
use serde_json::Value;
use wisent_onboarding_client::{JourneyError, ProductStarRequest, StarProduct, StarRequest};

#[derive(Debug, Subcommand)]
pub(super) enum StarCommand {
    /// Where the request stands: counting, shown or dismissed.
    Status,
    /// Never make the request again, until a reset.
    Dismiss,
    /// Count again from zero; set aside a journey file that holds no state.
    Reset,
}

fn request() -> Result<ProductStarRequest, JourneyError> {
    let product = StarProduct::from_package("ster", "Ster", env!("CARGO_PKG_REPOSITORY"))?;
    ProductStarRequest::new(product)
}

/// Count one run that completed: the request when this run makes it.
pub(crate) fn observed() -> Result<Option<StarRequest>, JourneyError> {
    request()?.observe()
}

/// Count one CLI run that completed, when a person reads this terminal. The
/// request, on the run that makes it, goes to standard error, so standard
/// output keeps only the answer; a request that cannot be kept is said there
/// too, and the run's own answer stands.
pub(super) fn observe() {
    match request() {
        Ok(request) => {
            request.observe_reporting();
        }
        Err(error) => {
            eprintln!("ster: this run is not counted toward the GitHub star request: {error}")
        }
    }
}

/// The status after `verb` (`status`, `dismiss` or `reset`), as the JSON
/// document `ster star` prints.
pub(crate) fn status_document(verb: &str) -> Result<Value, String> {
    let answered = request()
        .map_err(|error| error.to_string())
        .and_then(|request| request.command(&[verb.to_string()], true))?;
    serde_json::from_str(&answered)
        .map_err(|error| format!("the star status is not JSON ({error}): {answered}"))
}

/// `ster star status|dismiss|reset`: the status as JSON, or `path: value`
/// lines with `--text`.
pub(super) fn run(command: StarCommand) -> Result<()> {
    let verb = match command {
        StarCommand::Status => "status",
        StarCommand::Dismiss => "dismiss",
        StarCommand::Reset => "reset",
    };
    let document = status_document(verb).map_err(|refusal| anyhow!(refusal))?;
    super::answer_document(&document)
}
