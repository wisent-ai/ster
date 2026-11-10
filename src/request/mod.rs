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
//!   carries its `id`, `code`, `exit` (the status the same CLI command exits
//!   with), `usage`, `retryable` and `remedy`.
//! - two: the request was refused before anything ran: an unknown operation,
//!   a body that does not parse, or a field the operation's validation
//!   rejects. No log event precedes it, and `json` is
//!   `{"error": "<sentence>"}`.

use std::io::{self, Read, Write};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use ster::workflow;

mod jobs;
mod requests;
mod tuning;

use jobs::*;
use requests::*;

/// Completed request, as declared by the published NDJSON wire contract:
/// https://ster.wisent.com/docs/request-api#events-and-exit-status
const COMPLETED: i32 = 0;
/// Operational failure, as declared by the published NDJSON wire contract:
/// https://ster.wisent.com/docs/request-api#events-and-exit-status
const FAILED: i32 = 1;
/// Invalid request, as declared by the published NDJSON wire contract:
/// https://ster.wisent.com/docs/request-api#events-and-exit-status
const REFUSED: i32 = 2;

/// Run one request: read its body from stdin, run `operation`, print its
/// events on stdout, and return the status the process exits with.
pub fn run(operation: &str) -> Result<i32> {
    let mut body = Vec::new();
    io::stdin()
        .read_to_end(&mut body)
        .context("failed to read the request body from stdin")?;
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
    let counts = counted(operation);
    let job = Job { body, counts };
    if let Some(word) = operation.strip_prefix("tune/") {
        let request = tuning::decode(word, request_document(body))
            .map_err(|error| format!("{operation}: {error}"));
        return run_decoded_job(request, counts, crate::cli::tune::execute);
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
            match crate::cli::star::status_document(verb) {
                Ok(document) => {
                    emit_result(COMPLETED, document);
                    COMPLETED
                }
                Err(error) => {
                    emit_log(&format!("error: {error}\n"));
                    emit_result(FAILED, json!({"error": error}));
                    FAILED
                }
            }
        }
        "telemetry/status" | "telemetry/set" | "telemetry/show" | "telemetry/flush" => {
            let verb = operation.trim_start_matches("telemetry/");
            let body = match serde_json::from_slice::<Value>(request_document(body)) {
                Ok(body) => body,
                Err(error) => return refuse(&format!("{operation} body is not JSON: {error}")),
            };
            match crate::cli::telemetry::request_document(verb, &body) {
                Ok(document) => {
                    emit_result(COMPLETED, document);
                    COMPLETED
                }
                Err(error) => {
                    emit_log(&format!("error: {error}\n"));
                    emit_result(FAILED, json!({"error": error}));
                    FAILED
                }
            }
        }
        "errors/list" => job.run(errors_list_job),
        "errors/show" => job.run(errors_show_job),
        _ => refuse(&format!("unknown operation: {operation}")),
    }
}

/// A completed job may carry a failed domain report. Keep that report while
/// giving its refusal the same failed process status and log as a CLI refusal.
enum JobOutcome {
    Completed(Value),
    Failed {
        document: Value,
        refusal: anyhow::Error,
    },
}

impl From<Value> for JobOutcome {
    fn from(document: Value) -> Self {
        Self::Completed(document)
    }
}

/// One request's body, and whether its completion counts toward the GitHub
/// star request.
struct Job<'a> {
    body: &'a [u8],
    counts: bool,
}

impl Job<'_> {
    /// Parse and validate the body, then run the operation. Failures before
    /// it starts (a body that does not parse, a missing field) are refusals;
    /// a failure while it runs mirrors the CLI instead: the refusal sentence
    /// on a log event and one failed result.
    fn run<R, F, O>(&self, run: F) -> i32
    where
        R: DeserializeOwned + Validate,
        F: FnOnce(R) -> Result<O>,
        O: Into<JobOutcome>,
    {
        run_decoded_job(
            serde_json::from_slice::<R>(request_document(self.body)),
            self.counts,
            run,
        )
    }
}

fn request_document(body: &[u8]) -> &[u8] {
    if body.iter().all(u8::is_ascii_whitespace) {
        b"{}"
    } else {
        body
    }
}

fn run_decoded_job<R, F, O, E>(request: std::result::Result<R, E>, counts: bool, run: F) -> i32
where
    R: Validate,
    F: FnOnce(R) -> Result<O>,
    O: Into<JobOutcome>,
    E: std::fmt::Display,
{
    let request = match request {
        Ok(request) => request,
        Err(error) => {
            return refuse(&format!(
                "request body does not match the operation schema: {error}"
            ));
        }
    };
    if let Err(message) = request.validate() {
        return refuse(&message);
    }
    match run(request).map(Into::into) {
        Ok(JobOutcome::Completed(document)) => {
            if counts {
                star_request_event();
            }
            emit_result(COMPLETED, document);
            COMPLETED
        }
        Ok(JobOutcome::Failed { document, refusal }) => failed(document, &refusal),
        Err(error) => failed(json!({"error": format!("{error:#}")}), &error),
    }
}

/// A job that failed while it ran: its refusal on a log event, and `document`
/// as the failed result, carrying the declared cause when one refused it.
fn failed(mut document: Value, error: &anyhow::Error) -> i32 {
    emit_log(&format!("error: {error:#}\n"));
    if let (Some(declared), Some(fields)) =
        (ster::failures::declared(error), document.as_object_mut())
    {
        fields.insert("cause".to_owned(), declared.summary());
    }
    emit_result(FAILED, document);
    FAILED
}

fn refuse(message: &str) -> i32 {
    emit_result(REFUSED, json!({"error": message}));
    REFUSED
}

fn emit_log(chunk: &str) {
    emit(&json!({"type": "log", "stream": "stderr", "chunk": chunk}));
}

fn emit_result(status: i32, document: Value) {
    emit(&json!({"type": "result", "status": status, "json": document}));
}

/// One event per line, flushed when it is written. A reader that has gone
/// away ends the request: nobody is left to receive its result, and an
/// operation such as a training run would otherwise keep the machine busy
/// for a window that has already closed.
fn emit(event: &Value) {
    let mut out = io::stdout().lock();
    if let Err(error) = writeln!(out, "{event}") {
        eprintln!("Ster request cannot write its stdout event: {error}");
        std::process::exit(FAILED);
    }
    if let Err(error) = out.flush() {
        eprintln!("Ster request cannot flush its stdout event: {error}");
        std::process::exit(FAILED);
    }
}
