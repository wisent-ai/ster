//! `ster support report|status`: ask Ster's support for help with this
//! installation's facts attached, and read the replies. wisent-errors files
//! and reads (docs/support.md there) and declares how each fails; Ster Desktop
//! runs the same through `ster request support/<verb>`.

use anyhow::Result;
use clap::Subcommand;
use serde_json::{Value, json};
use ster::failures::{relay, telemetry as causes};
use wisent_errors::Failure;
use wisent_errors::telemetry::Report;

use super::telemetry::telemetry;

#[derive(Debug, Subcommand)]
pub(super) enum SupportCommand {
    /// File a report with Ster's support: what you were doing and what
    /// happened, with Ster's version, this system's version, this
    /// installation's id and, unless --without-failure, the last failure
    /// telemetry holds.
    Report {
        /// What you were doing and what happened.
        #[arg(long)]
        description: String,
        /// One line naming the problem.
        #[arg(long)]
        subject: Option<String>,
        /// Where support may answer; without it, read the reply with
        /// `ster support status`.
        #[arg(long)]
        email: Option<String>,
        /// Leave the last failure telemetry holds out of the report.
        #[arg(long)]
        without_failure: bool,
    },
    /// Every report this installation filed, with its state and the replies,
    /// as `{"reports": [...]}`.
    Status,
}

/// The refusal wisent-errors' support client declared and handed back as
/// `failure`, with `step`, what Ster was running, after its sentence.
fn refusal(failure: Failure, step: &str) -> anyhow::Error {
    relay(
        &failure,
        &causes::operations(),
        &causes::SUPPORT_UNDECLARED,
        step,
    )
}

pub(super) fn run(command: SupportCommand) -> Result<()> {
    match command {
        SupportCommand::Report {
            description,
            subject,
            email,
            without_failure,
        } => super::answer_document(&filed(
            &Report {
                subject: subject.as_deref(),
                description: &description,
                email: email.as_deref(),
                with_failure: !without_failure,
            },
            "`ster support report`",
        )?),
        SupportCommand::Status => super::answer_document(&status("`ster support status`")?),
    }
}

/// What `ster request support/report` answers: the report as filed, the
/// document `ster support report` prints.
pub(crate) fn report_document(report: &Report) -> Result<Value> {
    filed(report, "`ster request support/report`")
}

/// What `ster request support/status` answers, `{"reports": [...]}` as `ster
/// support status` prints it.
pub(crate) fn status_document() -> Result<Value> {
    status("`ster request support/status`")
}

/// File `report`; a refusal is wisent-errors' declared one, after `step`.
fn filed(report: &Report, step: &str) -> Result<Value> {
    telemetry()
        .file_report(report)
        .map_err(|failure| refusal(failure, step))
}

/// Every report this installation filed; a refusal is wisent-errors'
/// declared one, after `step`.
fn status(step: &str) -> Result<Value> {
    let reports = telemetry()
        .reports()
        .map_err(|failure| refusal(failure, step))?;
    Ok(json!({ "reports": reports }))
}
