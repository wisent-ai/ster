//! `ster telemetry`: what this installation reports to Ster's maintainers,
//! and the report itself. With the user's consent, a command that fails
//! queues its failure, and with `usage` every command queues its name, exit
//! status and duration; a detached `ster telemetry flush` sends the queue to
//! Echo's app collector as the command ends, so nobody waits for it. The
//! contract is wisent-errors' docs/telemetry.md; this is Ster's surface.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow};
use clap::Subcommand;
use serde_json::json;
use wisent_errors::telemetry::{Level, Product, Telemetry};
use wisent_errors::{Code, Failure};

/// Ster's command line, as Echo's app collector declares it. The release
/// build passes `WISENT_TELEMETRY_COLLECTOR_URL`, so only our release knows
/// our collector.
const PRODUCT: Product = Product {
    id: "ster",
    app_id: "ster-cli",
    version: env!("CARGO_PKG_VERSION"),
    built_collector: option_env!("WISENT_TELEMETRY_COLLECTOR_URL"),
};

#[derive(Debug, Subcommand)]
pub(super) enum TelemetryCommand {
    /// What this installation sends, where to, what waits, and how the last
    /// delivery went.
    Status,
    /// Send failures (`errors`), or failures and the commands that ran
    /// (`usage`), never their arguments.
    On {
        #[arg(long, value_parser = ["errors", "usage"])]
        level: String,
    },
    /// Send nothing, and drop every event still waiting.
    Off,
    /// Every event waiting to be sent, exactly as it will go.
    Show,
    /// Write every waiting event to FILE as one JSON array, for a machine
    /// without network.
    Export { file: PathBuf },
    /// Send every waiting event now.
    Flush,
}

pub(super) fn telemetry() -> Telemetry {
    Telemetry::new(PRODUCT)
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

pub(super) fn run(command: TelemetryCommand) -> Result<()> {
    let telemetry = telemetry();
    match command {
        TelemetryCommand::Status => {}
        TelemetryCommand::On { level } => {
            let level = Level::parse(&level, PRODUCT.id).map_err(refusal)?;
            telemetry.set_level(level).map_err(refusal)?;
        }
        TelemetryCommand::Off => telemetry.set_level(Level::Off).map_err(refusal)?,
        TelemetryCommand::Show => {
            let queued = telemetry.queued().map_err(refusal)?;
            return super::answer_document(&json!(queued));
        }
        TelemetryCommand::Export { file } => {
            let events = telemetry.export(&file).map_err(refusal)?;
            return super::answer_document(&json!({ "file": file, "events": events }));
        }
        TelemetryCommand::Flush => {
            let delivered = telemetry.flush().map_err(refusal)?;
            super::answer_document(&delivered.to_json())?;
            return match delivered.stopped {
                Some(stopped) => Err(refusal(stopped)),
                None => Ok(()),
            };
        }
    }
    super::answer_document(&telemetry.status())
}

/// The words of the command a person ran (`pairs import`), never its
/// arguments, and when it started.
pub(crate) struct Started {
    pub(crate) command: String,
    pub(crate) at: Instant,
}

/// Whether a command reports itself: `ster telemetry` and the desktop's
/// `ster request` do not, the first because it is the reporting, the second
/// because Ster Desktop reports what its requests do.
pub(super) fn reports(command: &str) -> bool {
    !(command.starts_with("telemetry") || command.starts_with("request"))
}

/// Ask once, on a terminal, what may be sent.
pub(super) fn ask() {
    telemetry().ask("Ster");
}

/// Queue what this run did, as the user agreed, and start the detached flush.
/// Nothing here changes the run's own output or exit status.
pub(crate) fn finish(started: &Started, error: Option<&anyhow::Error>, status: i32) {
    if !reports(&started.command) {
        return;
    }
    let telemetry = telemetry();
    if let Some(error) = error {
        match ster::failures::declared(error) {
            Some(declared) => telemetry.failure(&declared.failure(), Some(declared.cause.id)),
            None => {
                let point = format!("ster.cli.{}", started.command.replace(' ', "."));
                let failure =
                    Failure::or_fallback(point, Code::Unknown, "ster").detail(format!("{error:#}"));
                telemetry.failure(&failure, None);
            }
        }
    }
    telemetry.command(&started.command, status, started.at.elapsed());
    if let Ok(program) = std::env::current_exe() {
        telemetry.spawn_flush(&program, &["telemetry", "flush"]);
    }
}
