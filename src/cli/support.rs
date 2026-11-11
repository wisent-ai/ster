//! `ster support report|status`: ask Ster's support for help with this
//! installation's facts attached, and read the replies. wisent-errors files
//! and reads (docs/support.md there); Ster Desktop runs the same through
//! `ster request support/<verb>`.

use anyhow::{Result, anyhow};
use clap::Subcommand;
use serde_json::{Value, json};
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

fn refusal(failure: Failure) -> anyhow::Error {
    let detail = failure
        .detail
        .unwrap_or_else(|| failure.code.as_str().to_owned());
    match failure.remedy {
        Some(remedy) => anyhow!("{detail}; fix: {remedy}"),
        None => anyhow!(detail),
    }
}

pub(super) fn run(command: SupportCommand) -> Result<()> {
    let telemetry = telemetry();
    match command {
        SupportCommand::Report {
            description,
            subject,
            email,
            without_failure,
        } => {
            let filed = telemetry
                .file_report(&Report {
                    subject: subject.as_deref(),
                    description: &description,
                    email: email.as_deref(),
                    with_failure: !without_failure,
                })
                .map_err(refusal)?;
            super::answer_document(&filed)
        }
        SupportCommand::Status => {
            let reports = telemetry.reports().map_err(refusal)?;
            super::answer_document(&json!({ "reports": reports }))
        }
    }
}

/// What `ster request support/<verb>` answers, the document the CLI prints:
/// `report` with `{"description", "subject"?, "email"?, "with_failure"?}`
/// (with_failure defaults to true, as on the command line), and `status`,
/// `{"reports": [...]}`. A refusal is the sentence the CLI prints.
pub(crate) fn request_document(verb: &str, body: &Value) -> Result<Value, String> {
    let telemetry = telemetry();
    let said = |failure: Failure| refusal(failure).to_string();
    match verb {
        "report" => {
            let text = |name: &str| body.get(name).and_then(Value::as_str);
            let description = text("description").ok_or(
                "support/report needs {\"description\": \"what you were doing and what happened\"}",
            )?;
            telemetry
                .file_report(&Report {
                    subject: text("subject"),
                    description,
                    email: text("email"),
                    with_failure: body
                        .get("with_failure")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                })
                .map_err(said)
        }
        "status" => telemetry
            .reports()
            .map(|reports| json!({ "reports": reports }))
            .map_err(said),
        other => Err(format!("unknown operation: support/{other}")),
    }
}
