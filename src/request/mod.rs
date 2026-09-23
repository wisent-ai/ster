//! `ster request <operation>`: one JSON request from a desktop app, run to
//! completion by this process.
//!
//! Ster Desktop starts one of these per operation (`ster request train`,
//! `ster request pairs/inspect`, `ster request tune/sft`, and so on), writes
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
//!   event, and `json` is `{"error": "<sentence>"}`.
//! - two: the request was refused before anything ran: an unknown operation,
//!   a body that does not parse, or a field the operation's validation
//!   rejects. No log event precedes it, and `json` is
//!   `{"error": "<sentence>"}`.

use std::io::{self, Read, Write};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::workflow;

mod jobs;
mod requests;

use jobs::*;
use requests::*;

/// The operation started and failed.
const FAILED: i32 = 1;
/// The request was refused before the operation started.
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
    match operation {
        "workspace/import-pairs" => run_job(body, workspace_import_pairs_job),
        "train" => run_job(body, train_job),
        "optimize" => run_job(body, optimize_job),
        "evaluate" => run_job(body, evaluate_job),
        "generate" => run_job(body, generate_job),
        "extract" => run_job(body, extract_job),
        "inspect" => run_job(body, inspect_job),
        "decide" => run_job(body, decide_job),
        "calibrate" => run_job(body, calibrate_job),
        "pairs/inspect" => run_job(body, pairs_inspect_job),
        "pairs/save" => run_job(body, pairs_save_job),
        "pairs/synthesize" => run_job(body, pairs_synthesize_job),
        "tune/sft" => run_job(body, tune_sft_job),
        "tune/dpo" => run_job(body, tune_dpo_job),
        "tune/reward" => run_job(body, tune_reward_job),
        "tune/grpo" => run_job(body, tune_grpo_job),
        "tune/merge" => run_job(body, tune_merge_job),
        "tune/evaluate" => run_job(body, tune_evaluate_job),
        "tune/inspect" => run_job(body, tune_inspect_job),
        _ => refuse(&format!("unknown operation: {operation}")),
    }
}

/// Parse and validate the body, then run the operation. Failures before it
/// starts (a body that does not parse, a missing field) are refusals; a
/// failure while it runs mirrors the CLI instead: the refusal sentence on a
/// log event and one failed result.
fn run_job<R, F>(body: &[u8], run: F) -> i32
where
    R: DeserializeOwned + Validate,
    F: FnOnce(R) -> Result<Value>,
{
    let document = if body.iter().all(u8::is_ascii_whitespace) {
        b"{}".as_slice()
    } else {
        body
    };
    let request: R = match serde_json::from_slice(document) {
        Ok(request) => request,
        Err(error) => return refuse(&format!("request body is not valid JSON: {error}")),
    };
    if let Err(message) = request.validate() {
        return refuse(&message);
    }
    match run(request) {
        Ok(document) => {
            emit_result(0, document);
            0
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
    if writeln!(out, "{event}").and_then(|()| out.flush()).is_err() {
        std::process::exit(FAILED);
    }
}
