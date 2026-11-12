//! What this installation keeps for the person using it: the GitHub star
//! request (`star/status|dismiss|reset`), what it reports to Ster's
//! maintainers (`telemetry/status|set|show|flush`) and the reports it files
//! with Ster's support (`support/report|status`), the same documents `ster
//! star`, `ster telemetry` and `ster support` print.

use anyhow::Result;
use serde::Deserialize;

use super::{Validate, require};

/// `star/*`, `telemetry/status`, `telemetry/show`, `telemetry/flush` and
/// `support/status`: no fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::request) struct NoFieldsRequest {}

impl Validate for NoFieldsRequest {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// `telemetry/set`: the level to report at, as `ster telemetry on --level`
/// and `ster telemetry off` set it.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct TelemetrySetRequest {
    #[serde(default)]
    pub(in crate::request) level: String,
}

impl Validate for TelemetrySetRequest {
    fn validate(&self) -> Result<()> {
        require(
            &self.level,
            "telemetry/set",
            "level",
            "the level to report at: off, errors or usage",
            "--level of `ster telemetry on`, or `ster telemetry off`",
        )
    }
}

/// `support/report`: what `ster support report` files. `with_failure`
/// defaults to true, as the command line attaches the last failure unless
/// `--without-failure`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::request) struct SupportReportRequest {
    #[serde(default)]
    pub(in crate::request) description: String,
    #[serde(default)]
    pub(in crate::request) subject: Option<String>,
    #[serde(default)]
    pub(in crate::request) email: Option<String>,
    #[serde(default = "attach_failure")]
    pub(in crate::request) with_failure: bool,
}

/// The command line's default: the last failure goes with the report.
fn attach_failure() -> bool {
    true
}

impl Validate for SupportReportRequest {
    fn validate(&self) -> Result<()> {
        require(
            &self.description,
            "support/report",
            "description",
            "what you were doing and what happened",
            "--description of `ster support report`",
        )
    }
}
