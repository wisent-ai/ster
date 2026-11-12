//! The NDJSON events a request prints on stdout: its log lines and its one
//! result, completed, failed or refused.

use std::io::{self, Write};

use serde_json::{Value, json};

use ster::failures::{declared, exit_code, file, refuse as refused, request::EVENT_UNWRITTEN};

/// Completed request, as declared by the published NDJSON wire contract:
/// https://ster.wisent.com/docs/request-api#events-and-exit-status
pub(super) const COMPLETED: i32 = 0;
/// Operational failure, as declared by the published NDJSON wire contract:
/// https://ster.wisent.com/docs/request-api#events-and-exit-status
pub(super) const FAILED: i32 = 1;
/// Invalid request, as declared by the published NDJSON wire contract:
/// https://ster.wisent.com/docs/request-api#events-and-exit-status
pub(super) const REFUSED: i32 = 2;

/// A job that failed while it ran: its refusal on a log event, and `document`
/// as the failed result, carrying the declared cause when one refused it.
pub(super) fn failed(mut document: Value, error: &anyhow::Error) -> i32 {
    emit_log(&format!("error: {error:#}\n"));
    if let (Some(declared), Some(fields)) = (declared(error), document.as_object_mut()) {
        fields.insert("cause".to_owned(), declared.summary());
    }
    emit_result(FAILED, document);
    FAILED
}

/// A request refused before anything ran: its sentence and the cause that
/// declares it.
pub(super) fn refuse(refusal: &anyhow::Error) -> i32 {
    let mut document = json!({"error": format!("{refusal:#}")});
    if let (Some(declared), Some(fields)) = (declared(refusal), document.as_object_mut()) {
        fields.insert("cause".to_owned(), declared.summary());
    }
    emit_result(REFUSED, document);
    REFUSED
}

pub(super) fn emit_log(chunk: &str) {
    emit(&json!({"type": "log", "stream": "stderr", "chunk": chunk}));
}

pub(super) fn emit_result(status: i32, document: Value) {
    emit(&json!({"type": "result", "status": status, "json": document}));
}

/// One event per line, flushed when it is written. A reader that has gone
/// away ends the request with its declared refusal on stderr: nobody is left
/// to receive its result, and an operation such as a training run would
/// otherwise keep the machine busy for a window that has already closed.
pub(super) fn emit(event: &Value) {
    let mut out = io::stdout().lock();
    let written = writeln!(out, "{event}").and_then(|()| out.flush());
    if let Err(error) = written {
        let kind = event["type"].as_str().map_or("unnamed".to_owned(), str::to_owned);
        let refusal = refused(
            &EVENT_UNWRITTEN,
            &[
                ("event", &kind),
                ("kind", &file::kind(&error)),
                ("error", &error.to_string()),
            ],
        );
        eprintln!("Error: {refusal}");
        std::process::exit(exit_code(&refusal));
    }
}
