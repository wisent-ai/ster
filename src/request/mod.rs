//! `ster request <operation>`: one JSON request from a desktop app, run to
//! completion by this process.
//!
//! Ster Desktop starts one of these per operation (`ster request
//! vector/train`, `ster request pairs/inspect`, `ster request tune/sft`, and
//! so on), writes
//! the request body to stdin as one JSON document, and reads the events this
//! prints on stdout. The process ends when the operation ends, so nothing of
//! Ster stays resident between two operations. Every operation reuses the
//! exact functions the CLI commands use (workflow, runtime, artifact, tune,
//! lora): there is no parallel implementation.
//!
//! stdout carries only NDJSON events:
//!
//!   {"type":"log","stream":"stderr","chunk":"..."}      (zero or more)
//!   {"type":"star_request","title":"...","body":"...","repository":"...","actions":5}
//!                                                        (at most one, before the result)
//!   {"type":"result","status":<exit>,"json":{...}}      (exactly one, last)
//!
//! `star_request` is the GitHub star request the shared onboarding runtime
//! makes once, when a completed `vector/train`, `vector/optimize`,
//! `generate`, `decide` or `tune/<objective>` is the fifth counted run: the
//! desktop shows it beside the result. `star/status`, `star/dismiss` and
//! `star/reset` answer the same status `ster star` prints.
//!
//! `json` is the same document the CLI command prints, and `status` is the
//! status the process exits with:
//!
//! - zero: the operation completed.
//! - one: it failed while running. The refusal sentence is also the last log
//!   event. `json` is `{"error": "<sentence>"}`, or a rejected/conflicting
//!   workspace import's complete domain report with its actual refusal.
//!   When a declared cause refused it (`ster errors list`), `json.cause`
//!   carries the `functionality` that declares it, its `id`, `docs` (its
//!   line on the area page), `code`, `exit` (the status the same CLI command
//!   exits with), `usage`, `retryable` and `remedy`.
//! - two: the request was refused before anything ran: an unknown operation,
//!   a body that cannot be read (a directory as stdin, a broken stream) or
//!   does not parse, or a field the operation's validation rejects. No log
//!   event precedes it, and `json` is `{"error": "<sentence>", "cause":
//!   {...}}`, the cause `ster.request`'s or the operation's own. A standard
//!   output that reaches no reader (closed, or the null device) is refused
//!   before stdin is read, on stderr, with the same status 2.

use anyhow::Result;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use ster::workflow;

mod events;
mod jobs;
mod requests;
mod streams;
mod tuning;

use events::{COMPLETED, emit, emit_log, emit_result, failed, refuse};
use jobs::*;
use requests::*;

/// Every operation `ster request` serves, as [`dispatch`] matches them; an
/// unknown one is refused naming these.
const SERVED: &[&str] = &[
    "model/fixture/create",
    "model/tokenizer/create",
    "workspace/import",
    "workspace/show",
    "workspace/select",
    "workspace/remove",
    "vector/train",
    "vector/optimize",
    "vector/curve",
    "vector/ablate",
    "vector/evaluate",
    "vector/inspect",
    "vector/compare",
    "vector/project",
    "vector/score",
    "generate",
    "extract",
    "parity",
    "decide",
    "calibrate",
    "pairs/import",
    "pairs/inspect",
    "pairs/merge",
    "pairs/save",
    "pairs/synthesize",
    "star/status",
    "star/dismiss",
    "star/reset",
    "telemetry/status",
    "telemetry/set",
    "telemetry/show",
    "telemetry/flush",
    "telemetry/events",
    "doctor",
    "support/report",
    "support/status",
    "errors/list",
    "errors/show",
    "tune/sft",
    "tune/dpo",
    "tune/bipo",
    "tune/reward",
    "tune/grpo",
    "tune/decide",
    "tune/merge",
    "tune/export",
    "tune/evaluate",
    "tune/inspect",
];

/// Run one request: refuse it on stderr when its standard output reaches no
/// reader, read its body from stdin, run `operation`, print its events on
/// stdout, and return the status the process exits with.
pub fn run(operation: &str) -> Result<i32> {
    streams::answerable(operation)?;
    let body = match streams::body(operation) {
        Ok(body) => body,
        Err(refusal) => return Ok(refuse(&refusal)),
    };
    // Progress lines become log events, so the desktop's live log shows each
    // one when the workflow writes it rather than after the run.
    workflow::set_progress_sink(Some(Box::new(|line: &str| {
        emit_log(&format!("{line}\n"));
    })));
    let status = dispatch(operation, &body);
    workflow::set_progress_sink(None);
    Ok(status)
}

/// The operations whose completion counts toward the GitHub star request:
/// each runs a model and answers what a person asked for.
fn counted(operation: &str) -> bool {
    matches!(
        operation,
        "vector/train"
            | "vector/optimize"
            | "generate"
            | "decide"
            | "tune/sft"
            | "tune/dpo"
            | "tune/bipo"
            | "tune/reward"
            | "tune/grpo"
            | "tune/decide"
    )
}

/// The star request, as its event, when this completed run makes it; said on
/// a log event when the request cannot be kept, and the run's own result
/// stands.
fn star_request_event() {
    match crate::cli::star::observed() {
        Ok(Some(request)) => emit(&json!({
            "type": "star_request",
            "title": request.title,
            "body": request.body,
            "repository": request.repository,
            "actions": request.actions,
        })),
        Ok(None) => {}
        Err(cause) => emit_log(&format!(
            "ster: this run is not counted toward the GitHub star request: {cause}\n"
        )),
    }
}

fn dispatch(operation: &str, body: &[u8]) -> i32 {
    if !SERVED.contains(&operation) {
        return unknown(operation);
    }
    let counts = counted(operation);
    let job = Job {
        operation,
        body,
        counts,
    };
    if let Some(word) = operation.strip_prefix("tune/") {
        let request = tuning::decode(word, request_document(body));
        return run_decoded_job(
            operation,
            body.len(),
            request,
            counts,
            crate::cli::tune::execute,
        );
    }
    match operation {
        "model/fixture/create" => job.run(fixture_create_job),
        "model/tokenizer/create" => job.run(tokenizer_create_job),
        "workspace/import" => job.run(workspace_import_pairs_job),
        "workspace/show" => job.run(workspace_show_job),
        "workspace/select" => job.run(workspace_select_job),
        "workspace/remove" => job.run(workspace_remove_job),
        "vector/train" => job.run(train_job),
        "vector/optimize" => job.run(optimize_job),
        "vector/curve" => job.run(curve_job),
        "vector/ablate" => job.run(ablate_job),
        "vector/evaluate" => job.run(evaluate_job),
        "vector/inspect" => job.run(inspect_job),
        "vector/compare" => job.run(compare_job),
        "vector/project" => job.run(project_job),
        "vector/score" => job.run(score_job),
        "generate" => job.run(generate_job),
        "extract" => job.run(extract_job),
        "parity" => job.run(parity_job),
        "decide" => job.run(decide_job),
        "calibrate" => job.run(calibrate_job),
        "pairs/import" => job.run(pairs_import_job),
        "pairs/inspect" => job.run(pairs_inspect_job),
        "pairs/merge" => job.run(pairs_merge_job),
        "pairs/save" => job.run(pairs_save_job),
        "pairs/synthesize" => job.run(pairs_synthesize_job),
        "star/status" | "star/dismiss" | "star/reset" => {
            let verb = operation.trim_start_matches("star/");
            job.run(|_: NoFieldsRequest| crate::cli::star::status_document(verb))
        }
        "telemetry/set" => job.run(|request: TelemetrySetRequest| {
            crate::cli::telemetry::request_document("set", Some(&request.level))
        }),
        "telemetry/status" | "telemetry/show" | "telemetry/flush" | "telemetry/events" => {
            let verb = operation.trim_start_matches("telemetry/");
            job.run(|_: NoFieldsRequest| crate::cli::telemetry::request_document(verb, None))
        }
        // Completes with the findings even when a functionality fails, so the
        // desktop's Diagnosis screen shows every finding.
        "doctor" => job.run(|_: NoFieldsRequest| Ok(crate::cli::doctor::document())),
        "support/report" => job.run(|request: SupportReportRequest| {
            crate::cli::support::report_document(&wisent_errors::telemetry::Report {
                subject: request.subject.as_deref(),
                description: &request.description,
                email: request.email.as_deref(),
                with_failure: request.with_failure,
            })
        }),
        "support/status" => job.run(|_: NoFieldsRequest| crate::cli::support::status_document()),
        "errors/list" => job.run(errors_list_job),
        "errors/show" => job.run(errors_show_job),
        _ => unknown(operation),
    }
}

/// The refusal of an operation `ster request` does not serve.
fn unknown(operation: &str) -> i32 {
    refuse(&ster::failures::refuse(
        &ster::failures::request::OPERATION_UNKNOWN,
        &[("operation", operation), ("served", &SERVED.join(", "))],
    ))
}

/// One request's body, and whether its completion counts toward the GitHub
/// star request.
struct Job<'a> {
    operation: &'a str,
    body: &'a [u8],
    counts: bool,
}

mod job;

use job::{request_document, run_decoded_job};
