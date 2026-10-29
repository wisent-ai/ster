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
//!   {"type":"result","status":<exit>,"json":{...}}      (exactly one, last)
//!
//! `json` is the same document the CLI command prints, and `status` is the
//! status the process exits with:
//!
//! - zero: the operation completed.
//! - one: it failed while running. The refusal sentence is also the last log
//!   event. `json` is `{"error": "<sentence>"}`, or a rejected/conflicting
//!   workspace import's complete domain report with its actual refusal.
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

fn dispatch(operation: &str, body: &[u8]) -> i32 {
    if let Some(word) = operation.strip_prefix("tune/") {
        let request = tuning::decode(word, request_document(body))
            .map_err(|error| format!("{operation}: {error}"));
        return run_decoded_job(request, crate::cli::tune::execute);
    }
    match operation {
        "model/fixture/create" => run_job(body, fixture_create_job),
        "model/tokenizer/create" => run_job(body, tokenizer_create_job),
        "workspace/import" => run_job(body, workspace_import_pairs_job),
        "workspace/show" => run_job(body, workspace_show_job),
        "workspace/select" => run_job(body, workspace_select_job),
        "workspace/remove" => run_job(body, workspace_remove_job),
        "vector/train" => run_job(body, train_job),
        "vector/optimize" => run_job(body, optimize_job),
        "vector/curve" => run_job(body, curve_job),
        "vector/ablate" => run_job(body, ablate_job),
        "vector/evaluate" => run_job(body, evaluate_job),
        "vector/inspect" => run_job(body, inspect_job),
        "vector/compare" => run_job(body, compare_job),
        "vector/project" => run_job(body, project_job),
        "generate" => run_job(body, generate_job),
        "extract" => run_job(body, extract_job),
        "parity" => run_job(body, parity_job),
        "decide" => run_job(body, decide_job),
        "calibrate" => run_job(body, calibrate_job),
        "pairs/import" => run_job(body, pairs_import_job),
        "pairs/inspect" => run_job(body, pairs_inspect_job),
        "pairs/merge" => run_job(body, pairs_merge_job),
        "pairs/save" => run_job(body, pairs_save_job),
        "pairs/synthesize" => run_job(body, pairs_synthesize_job),
        _ => refuse(&format!("unknown operation: {operation}")),
    }
}

/// A completed job may carry a failed domain report. Keep that report while
/// giving its refusal the same failed process status and log as a CLI refusal.
enum JobOutcome {
    Completed(Value),
    Failed { document: Value, reason: String },
}

impl From<Value> for JobOutcome {
    fn from(document: Value) -> Self {
        Self::Completed(document)
    }
}

/// Parse and validate the body, then run the operation. Failures before it
/// starts (a body that does not parse, a missing field) are refusals; a
/// failure while it runs mirrors the CLI instead: the refusal sentence on a
/// log event and one failed result.
fn run_job<R, F, O>(body: &[u8], run: F) -> i32
where
    R: DeserializeOwned + Validate,
    F: FnOnce(R) -> Result<O>,
    O: Into<JobOutcome>,
{
    run_decoded_job(serde_json::from_slice::<R>(request_document(body)), run)
}

fn request_document(body: &[u8]) -> &[u8] {
    if body.iter().all(u8::is_ascii_whitespace) {
        b"{}"
    } else {
        body
    }
}

fn run_decoded_job<R, F, O, E>(request: std::result::Result<R, E>, run: F) -> i32
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
            emit_result(COMPLETED, document);
            COMPLETED
        }
        Ok(JobOutcome::Failed { document, reason }) => {
            emit_log(&format!("error: {reason}\n"));
            emit_result(FAILED, document);
            FAILED
        }
        Err(error) => {
            let message = format!("{error:#}");
            emit_log(&format!("error: {message}\n"));
            emit_result(FAILED, json!({"error": message}));
            FAILED
        }
    }
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
