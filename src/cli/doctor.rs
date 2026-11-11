//! `ster doctor`: every functionality Ster's failure catalogue declares with
//! an entry point, tried now on this machine, each said to work with what was
//! seen, to fail with what failed and its fix, or to be skipped and why. The
//! contract is wisent-errors' docs/diagnosis.md; the list is the catalogue's,
//! held to it by tests/doctor.

use anyhow::Result;
use wisent_errors::diagnosis::{self, Check, Probe};
use wisent_errors::{Code, Failure, Operation};

/// The functionality the catalogue declares under `id`.
fn operation(id: &str) -> &'static Operation {
    ster::failures::operations()
        .into_iter()
        .find(|operation| operation.id == id)
        .unwrap_or_else(|| panic!("the catalogue declares no {id}; tests/doctor holds the checks to it"))
}

/// A failure the way the command that runs `point` would raise it.
fn failure_of(point: &str, error: &anyhow::Error) -> Failure {
    match ster::failures::declared(error) {
        Some(declared) => declared.failure(),
        None => Failure::or_fallback(point, Code::Unknown, "ster").detail(format!("{error:#}")),
    }
}

/// Every functionality with an entry point and how the diagnosis tries it.
pub(crate) fn checks() -> Vec<Check> {
    vec![
        Check { operation: operation("ster.errors"), probe: Probe::Run(errors) },
        Check { operation: operation("ster.telemetry"), probe: Probe::Run(telemetry) },
        Check { operation: operation("ster.support"), probe: Probe::Run(support) },
        Check {
            operation: operation("ster.workspace.import"),
            probe: Probe::Skipped("it copies a pair set the user names into the workspace"),
        },
        Check { operation: operation("ster.workspace.show"), probe: Probe::Run(show) },
        Check {
            operation: operation("ster.workspace.select"),
            probe: Probe::Skipped("it changes which pair set is active"),
        },
        Check {
            operation: operation("ster.workspace.remove"),
            probe: Probe::Skipped("it removes a pair set from the workspace"),
        },
        Check { operation: operation("ster.workspace.active"), probe: Probe::Run(active) },
        Check { operation: operation("wisent-errors.telemetry.consent"), probe: Probe::Run(consent) },
        Check { operation: operation("wisent-errors.telemetry.record"), probe: Probe::Run(record) },
        Check {
            operation: operation("wisent-errors.telemetry.deliver"),
            probe: Probe::Skipped("it sends the events that wait; `ster telemetry flush` does, and ster.telemetry checks where they go"),
        },
        Check {
            operation: operation("wisent-errors.support.report"),
            probe: Probe::Skipped("it files a report with Ster's support"),
        },
        Check { operation: operation("wisent-errors.support.status"), probe: Probe::Run(replies) },
    ]
}

fn errors() -> Result<String, Failure> {
    let operations = ster::failures::operations();
    let problems = wisent_errors::catalogue_problems(&operations);
    if problems.is_empty() {
        Ok(format!("the catalogue declares {} functionalities and holds together", operations.len()))
    } else {
        Err(Failure::or_fallback("ster.errors", Code::Unknown, "ster")
            .detail(format!("the failure catalogue this build carries does not hold together: {}", problems.join("; ")))
            .remedy("install a released ster: `stado product update ster --surface cli`"))
    }
}

fn telemetry() -> Result<String, Failure> {
    let telemetry = super::telemetry::telemetry();
    let Some(directory) = telemetry.directory() else {
        return Err(Failure::or_fallback("ster.telemetry", Code::Config, "ster")
            .detail("no directory for telemetry: neither WISENT_TELEMETRY_HOME, XDG_STATE_HOME nor HOME is set")
            .remedy("set HOME, or WISENT_TELEMETRY_HOME to a directory ster may write"));
    };
    let probe = directory.join(".doctor");
    std::fs::create_dir_all(directory)
        .and_then(|()| std::fs::write(&probe, b"doctor"))
        .and_then(|()| std::fs::remove_file(&probe))
        .map_err(|error| {
            Failure::or_fallback("ster.telemetry", Code::Config, "ster")
                .detail(format!("writing in {}: {error}", directory.display()))
                .remedy(format!("make {} writable for this user, or set WISENT_TELEMETRY_HOME to one that is", directory.display()))
        })?;
    let collector = telemetry.collector()?;
    Ok(format!("{} can be written; events go to {}", directory.display(), collector.url))
}

fn support() -> Result<String, Failure> {
    let (address, source) = super::telemetry::telemetry().support_address()?;
    Ok(format!("reports go to {address} ({source})"))
}

fn replies() -> Result<String, Failure> {
    let reports = super::telemetry::telemetry().reports()?;
    Ok(match reports.len() {
        0 => "no report was filed from here".to_owned(),
        filed => format!("the replies to {filed} report(s) filed from here were read back"),
    })
}

fn consent() -> Result<String, Failure> {
    Ok(match super::telemetry::telemetry().level()? {
        Some(level) => format!("the user chose {}", level.as_str()),
        None => "the user has not been asked yet; nothing is queued until they answer".to_owned(),
    })
}

fn record() -> Result<String, Failure> {
    let waiting = super::telemetry::telemetry().queued()?;
    Ok(format!("the queue reads; {} event(s) wait", waiting.len()))
}

fn show() -> Result<String, Failure> {
    let summary = ster::workspace::summary().map_err(|error| failure_of("ster.workspace.show", &error))?;
    Ok(format!(
        "the workspace holds {} pair set(s), active: {}",
        summary.pair_sets.len(),
        summary.active_pair_set.as_deref().unwrap_or("none")
    ))
}

fn active() -> Result<String, Failure> {
    let path = ster::workspace::active_pair_set().map_err(|error| failure_of("ster.workspace.active", &error))?;
    Ok(format!("the active pair set {} is there", path.display()))
}

/// Print the findings, as lines or `--json`, and exit 1 when one failed.
pub(super) fn run(json: bool) -> Result<()> {
    let findings = diagnosis::diagnose(&checks());
    if json {
        println!("{}", serde_json::to_string_pretty(&diagnosis::to_json(&findings))?);
    } else {
        println!("{}", diagnosis::render(&findings));
    }
    let status = diagnosis::exit_status(&findings);
    if status != 0 {
        std::process::exit(status);
    }
    Ok(())
}
