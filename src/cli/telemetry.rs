//! `ster telemetry`: what this installation reports to Ster's maintainers,
//! and the report itself. With the user's consent, a command that fails
//! queues its failure, and with `usage` every command queues its name, exit
//! status and duration; a detached `ster telemetry flush` sends the queue to
//! Echo's app collector as the command ends, so nobody waits for it. The
//! contract is wisent-errors' docs/telemetry.md; this is Ster's surface.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Subcommand;
use serde_json::{Value, json};
use ster::failures::{relay, telemetry as causes};
use wisent_errors::telemetry::{Level, Product, Telemetry};
use wisent_errors::{Code, Failure};

/// Ster's command line, as Echo's app collector declares it. Events go to
/// Wisent's public collector unless `WISENT_TELEMETRY_COLLECTOR_URL` names
/// another.
const PRODUCT: Product = Product {
    id: "ster",
    app_id: "ster-cli",
    version: env!("CARGO_PKG_VERSION"),
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
    /// Every event waiting to be sent, exactly as it will go, as
    /// `{"events": [...]}`.
    Show,
    /// Write every waiting event to FILE as one JSON array, for a machine
    /// without network.
    Export { file: PathBuf },
    /// Send every waiting event now.
    Flush,
    /// Every kind of event Ster can send, the consent each needs and every
    /// property it carries: what the privacy page shows.
    Events,
}

pub(super) fn telemetry() -> Telemetry {
    Telemetry::new(PRODUCT)
}

/// The refusal wisent-errors' telemetry declared and handed back as
/// `failure`, with `step`, what Ster was running, after its sentence.
fn refusal(failure: Failure, step: &str) -> anyhow::Error {
    relay(&failure, &causes::operations(), &causes::UNDECLARED, step)
}

pub(super) fn run(command: TelemetryCommand) -> Result<()> {
    let telemetry = telemetry();
    match command {
        TelemetryCommand::Status => {}
        TelemetryCommand::On { level } => {
            let step = format!("`ster telemetry on --level {level}`");
            let level =
                Level::parse(&level, PRODUCT.id).map_err(|failure| refusal(failure, &step))?;
            telemetry
                .set_level(level)
                .map_err(|failure| refusal(failure, &step))?;
        }
        TelemetryCommand::Off => telemetry
            .set_level(Level::Off)
            .map_err(|failure| refusal(failure, "`ster telemetry off`"))?,
        TelemetryCommand::Show => {
            let queued = telemetry
                .queued()
                .map_err(|failure| refusal(failure, "`ster telemetry show`"))?;
            return super::answer_document(&json!({ "events": queued }));
        }
        TelemetryCommand::Export { file } => {
            let step = format!("`ster telemetry export {}`", file.display());
            let events = telemetry
                .export(&file)
                .map_err(|failure| refusal(failure, &step))?;
            return super::answer_document(&json!({ "file": file, "events": events }));
        }
        TelemetryCommand::Flush => {
            let step = "`ster telemetry flush`";
            let delivered = telemetry
                .flush()
                .map_err(|failure| refusal(failure, step))?;
            super::answer_document(&delivered.to_json())?;
            return match delivered.stopped {
                Some(stopped) => Err(refusal(stopped, step)),
                None => Ok(()),
            };
        }
        TelemetryCommand::Events => return super::answer_document(&events_document()),
    }
    super::answer_document(&telemetry.status())
}

/// wisent-errors' catalogue of every event, as `ster telemetry events`
/// prints it: the queue takes only these, with exactly these properties.
pub(crate) fn events_document() -> serde_json::Value {
    json!(
        wisent_errors::telemetry::EVENTS
            .iter()
            .map(|event| json!({
                "name": event.name,
                "consent": event.consent.as_str(),
                "summary": event.summary,
                "properties": event.properties,
            }))
            .collect::<Vec<_>>()
    )
}

/// What `ster request telemetry/<verb>` answers, the same document the CLI
/// prints: `status`; `set` to `level` (`off`, `errors` or `usage`), then the
/// status; `show`, the waiting events as `{"events": [...]}`, the object a
/// request result must be; `flush`, what it sent; `events`, every kind of
/// event Ster can send, as `{"events": [...]}`. A refusal, or a flush that
/// stopped, is the declared refusal the CLI exits with.
pub(crate) fn request_document(verb: &str, level: Option<&str>) -> Result<Value> {
    let telemetry = telemetry();
    let step = format!("`ster request telemetry/{verb}`");
    match (verb, level) {
        ("set", Some(level)) => {
            let level =
                Level::parse(level, PRODUCT.id).map_err(|failure| refusal(failure, &step))?;
            telemetry
                .set_level(level)
                .map_err(|failure| refusal(failure, &step))?;
            Ok(telemetry.status())
        }
        ("show", _) => telemetry
            .queued()
            .map(|queued| json!({ "events": queued }))
            .map_err(|failure| refusal(failure, &step)),
        ("flush", _) => {
            let delivered = telemetry
                .flush()
                .map_err(|failure| refusal(failure, &step))?;
            match delivered.stopped {
                Some(stopped) => Err(refusal(stopped, &step)),
                None => Ok(delivered.to_json()),
            }
        }
        // An object, as every request result is read: the CLI prints the list.
        ("events", _) => Ok(json!({ "events": events_document() })),
        _ => Ok(telemetry.status()),
    }
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
/// Nothing here changes the run's exit status; a flush that cannot start is
/// said on standard error, and its events stay queued for the next run.
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
    match std::env::current_exe() {
        Ok(program) => telemetry.spawn_flush(&program, &["telemetry", "flush"]),
        Err(error) => eprintln!(
            "ster: {}",
            ster::failures::refuse(
                &causes::PROGRAM_UNNAMED,
                &[
                    ("command", &started.command),
                    ("kind", &ster::failures::file::kind(&error)),
                    ("error", &error.to_string()),
                ],
            )
        ),
    }
}
