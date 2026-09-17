//! serve.rs — `ster serve`: loopback HTTP/JSON backend for desktop apps.
//!
//! The desktop app spawns this once (`ster serve --port 0`) and talks to it
//! over 127.0.0.1 HTTP — it never builds argv for the other CLI commands.
//! On bind, exactly one line lands on stdout:
//!
//!   {"ready":true,"port":<number>}
//!
//! After that, stdout carries no protocol traffic; every failure is an HTTP
//! response. All endpoints live under /v1 and every handler reuses the exact
//! functions the CLI commands use (workflow.rs, runtime.rs, artifact.rs,
//! tune.rs, lora.rs) — no parallel implementation.
//!
//! Errors are non-2xx with body {"error": "<one sentence>"} — the product's
//! own refusal sentence, verbatim from the underlying failure.
//!
//! Long-running jobs (every workflow) stream NDJSON:
//!   {"type":"log","stream":"stderr","chunk":"..."}   (zero or more)
//!   {"type":"result","status":0,"json":{...}}        (exactly one, last)
//! where `json` is the same document the CLI prints and `status` mirrors the
//! CLI exit code.
//!
//! The crate has no HTTP dependency, so this is a minimal std::net server:
//! one request per connection, responses close the connection.

use std::{
    io::{BufRead, BufReader, Read},
    net::{TcpListener, TcpStream},

    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::workflow;

const MAX_BODY_BYTES: usize = 1024 * 1024;

/// The shared write half of one connection: the handler writes the response
/// head and result events through it, and the progress sink writes log events
/// through a second handle on the same lock.
type SharedWriter = Arc<Mutex<TcpStream>>;

/// Start the serve backend. Binds 127.0.0.1 on `port` (0 = ephemeral),
/// prints the ready line, then serves until killed.
pub fn run(port: u16) -> Result<()> {
    let listener =
        TcpListener::bind(("127.0.0.1", port)).context("failed to bind the serve port")?;
    let bound = listener.local_addr()?.port();
    println!("{}", json!({"ready": true, "port": bound}));

    // Streamed jobs share one progress sink, so they run one at a time —
    // this keeps each job's log events on its own response.
    let job_lock = Arc::new(Mutex::new(()));
    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let job_lock = Arc::clone(&job_lock);
                thread::spawn(move || {
                    let _ = handle_connection(stream, &job_lock);
                });
            }
            Err(error) => eprintln!("serve accept failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection(stream: TcpStream, job_lock: &Mutex<()>) -> Result<()> {
    let writer: SharedWriter = Arc::new(Mutex::new(stream.try_clone()?));
    let mut reader = BufReader::new(stream);

    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        send_error(&writer, 400, "malformed request line");
        return Ok(());
    };
    let path = target.split(['?', '#']).next().unwrap_or(target);

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    if content_length > MAX_BODY_BYTES {
        send_error(&writer, 400, "request body too large");
        return Ok(());
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

    match (method, path) {
        ("GET", "/v1/health") => send_json(&writer, 200, json!({"status": "ok"})),
        ("GET", "/v1/workspace") => match crate::workspace::summary() {
            Ok(summary) => send_json(&writer, 200, serde_json::to_value(summary)?),
            Err(error) => send_error(&writer, 500, &format!("{error:#}")),
        },
        ("POST", "/v1/workspace/import-pairs") => {
            stream_job(&writer, &body, job_lock, workspace_import_pairs_job)
        }
        ("POST", "/v1/train") => stream_job(&writer, &body, job_lock, train_job),
        ("POST", "/v1/optimize") => stream_job(&writer, &body, job_lock, optimize_job),
        ("POST", "/v1/evaluate") => stream_job(&writer, &body, job_lock, evaluate_job),
        ("POST", "/v1/generate") => stream_job(&writer, &body, job_lock, generate_job),
        ("POST", "/v1/extract") => stream_job(&writer, &body, job_lock, extract_job),
        ("POST", "/v1/inspect") => stream_job(&writer, &body, job_lock, inspect_job),
        ("POST", "/v1/decide") => stream_job(&writer, &body, job_lock, decide_job),
        ("POST", "/v1/calibrate") => stream_job(&writer, &body, job_lock, calibrate_job),
        ("POST", "/v1/pairs/inspect") => stream_job(&writer, &body, job_lock, pairs_inspect_job),
        ("POST", "/v1/pairs/save") => stream_job(&writer, &body, job_lock, pairs_save_job),
        ("POST", "/v1/pairs/synthesize") => {
            stream_job(&writer, &body, job_lock, pairs_synthesize_job)
        }
        ("POST", "/v1/tune/sft") => stream_job(&writer, &body, job_lock, tune_sft_job),
        ("POST", "/v1/tune/dpo") => stream_job(&writer, &body, job_lock, tune_dpo_job),
        ("POST", "/v1/tune/reward") => stream_job(&writer, &body, job_lock, tune_reward_job),
        ("POST", "/v1/tune/grpo") => stream_job(&writer, &body, job_lock, tune_grpo_job),
        ("POST", "/v1/tune/merge") => stream_job(&writer, &body, job_lock, tune_merge_job),
        ("POST", "/v1/tune/evaluate") => stream_job(&writer, &body, job_lock, tune_evaluate_job),
        ("POST", "/v1/tune/inspect") => stream_job(&writer, &body, job_lock, tune_inspect_job),
        _ => send_error(&writer, 404, &format!("unknown endpoint: {method} {path}")),
    }
    Ok(())
}

/// Read + validate the request, then stream one job as NDJSON. Failures
/// before streaming (bad body, missing fields) are non-2xx error envelopes;
/// failures mid-job mirror the CLI instead: the refusal sentence on a stderr
/// log event plus one status-1 result.
fn stream_job<R, F>(writer: &SharedWriter, body: &[u8], job_lock: &Mutex<()>, run: F)
where
    R: DeserializeOwned + Validate,
    F: FnOnce(R) -> Result<Value>,
{
    let document = if body.iter().all(|byte| byte.is_ascii_whitespace()) {
        b"{}".as_slice()
    } else {
        body
    };
    let request: R = match serde_json::from_slice(document) {
        Ok(request) => request,
        Err(_) => {
            send_error(writer, 400, "request body is not valid JSON");
            return;
        }
    };
    if let Err(message) = request.validate() {
        send_error(writer, 400, &message);
        return;
    }

    write_response_head(writer, 200, "application/x-ndjson", None);
    let _job = job_lock.lock().expect("job lock");
    let sink_writer = Arc::clone(writer);
    workflow::set_progress_sink(Some(Box::new(move |line| {
        emit_log(&sink_writer, "stderr", &format!("{line}\n"));
    })));
    let outcome = run(request);
    workflow::set_progress_sink(None);
    match outcome {
        Ok(document) => emit_result(writer, 0, document),
        Err(error) => {
            let message = format!("{error:#}");
            emit_log(writer, "stderr", &format!("error: {message}\n"));
            emit_result(writer, 1, json!({"error": message}));
        }
    }
}

mod jobs;
mod requests;
mod response;

use jobs::*;
use requests::*;
use response::*;
