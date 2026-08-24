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
//! functions the CLI commands use (workflow.rs, runtime.rs, artifact.rs) — no
//! parallel implementation.
//!
//! Errors are non-2xx with body {"error": "<one sentence>"} — the product's
//! own refusal sentence, verbatim from the underlying failure.
//!
//! Long-running jobs (all six workflows) stream NDJSON:
//!   {"type":"log","stream":"stderr","chunk":"..."}   (zero or more)
//!   {"type":"result","status":0,"json":{...}}        (exactly one, last)
//! where `json` is the same document the CLI prints and `status` mirrors the
//! CLI exit code.
//!
//! The crate has no HTTP dependency, so this is a minimal std::net server:
//! one request per connection, responses close the connection.

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Context, Result};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{
    DeviceChoice, GenerationOptions, PairSet, Runtime, SteeringArtifact, TrainingMethod,
    workflow::{self, parse_layers},
};

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
        ("POST", "/v1/train") => stream_job(&writer, &body, job_lock, train_job),
        ("POST", "/v1/optimize") => stream_job(&writer, &body, job_lock, optimize_job),
        ("POST", "/v1/evaluate") => stream_job(&writer, &body, job_lock, evaluate_job),
        ("POST", "/v1/generate") => stream_job(&writer, &body, job_lock, generate_job),
        ("POST", "/v1/extract") => stream_job(&writer, &body, job_lock, extract_job),
        ("POST", "/v1/inspect") => stream_job(&writer, &body, job_lock, inspect_job),
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

// MARK: - Requests

/// Field-level validation before a job starts streaming. The message is the
/// one-sentence refusal the desktop shows for a malformed request.
trait Validate {
    fn validate(&self) -> Result<(), String>;
}

fn require(value: &str, sentence: String) -> Result<(), String> {
    if value.trim().is_empty() { Err(sentence) } else { Ok(()) }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default = "default_device")]
    device: String,
}

impl ModelRequest {
    fn check(&self, action: &str) -> Result<(), String> {
        require(&self.model, format!("{action} requires a model"))
    }

    fn load_runtime(&self) -> Result<Runtime> {
        let device = DeviceChoice::parse(&self.device)?;
        Runtime::load(&self.model, self.revision.as_deref(), device)
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainRequest {
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default)]
    pairs: String,
    #[serde(default)]
    output: String,
    #[serde(default = "default_layers")]
    layers: String,
    #[serde(default = "default_method")]
    method: String,
}

impl Validate for TrainRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("train")?;
        require(&self.pairs, "train requires a pairs file".to_owned())?;
        require(&self.output, "train requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OptimizeRequest {
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default)]
    pairs: String,
    #[serde(default)]
    output: String,
    #[serde(default = "default_layers")]
    layers: String,
}

impl Validate for OptimizeRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("optimize")?;
        require(&self.pairs, "optimize requires a pairs file".to_owned())?;
        require(&self.output, "optimize requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvaluateRequest {
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default)]
    pairs: String,
    #[serde(default)]
    vector: String,
}

impl Validate for EvaluateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("evaluate")?;
        require(&self.pairs, "evaluate requires a pairs file".to_owned())?;
        require(&self.vector, "evaluate requires a steering artifact".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateRequest {
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    vector: Option<String>,
    #[serde(default = "default_strength")]
    strength: f64,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default)]
    temperature: f64,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default = "default_seed")]
    seed: u64,
}

impl Validate for GenerateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("generate")?;
        require(&self.prompt, "generate requires a prompt".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExtractRequest {
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default)]
    input: String,
    #[serde(default)]
    output: String,
    #[serde(default = "default_layers")]
    layers: String,
}

impl Validate for ExtractRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("extract")?;
        require(&self.input, "extract requires a prompt input file".to_owned())?;
        require(&self.output, "extract requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InspectRequest {
    #[serde(default)]
    artifact: String,
}

impl Validate for InspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.artifact, "inspect requires a steering artifact".to_owned())
    }
}

fn default_device() -> String {
    "cpu".to_owned()
}

fn default_layers() -> String {
    "all".to_owned()
}

fn default_method() -> String {
    "caa".to_owned()
}

fn default_strength() -> f64 {
    1.0
}

fn default_max_new_tokens() -> usize {
    128
}

fn default_seed() -> u64 {
    42
}

// MARK: - Jobs

/// Every job mirrors its CLI arm in main.rs: same loads, same workflow call,
/// and the returned document is the same payload the CLI prints.
fn train_job(request: TrainRequest) -> Result<Value> {
    let runtime = request.model.load_runtime()?;
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let method = TrainingMethod::parse(&request.method)?;
    let artifact = workflow::train(&runtime, &pair_set, &layers, method)?;
    artifact.save(Path::new(&request.output))?;
    Ok(workflow::artifact_summary(&artifact))
}

fn optimize_job(request: OptimizeRequest) -> Result<Value> {
    let runtime = request.model.load_runtime()?;
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let artifact = workflow::optimize(&runtime, &pair_set, &layers)?;
    artifact.save(Path::new(&request.output))?;
    Ok(workflow::artifact_summary(&artifact))
}

fn evaluate_job(request: EvaluateRequest) -> Result<Value> {
    let runtime = request.model.load_runtime()?;
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let artifact = SteeringArtifact::load(Path::new(&request.vector))?;
    let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
    Ok(serde_json::to_value(&report)?)
}

fn generate_job(request: GenerateRequest) -> Result<Value> {
    let runtime = request.model.load_runtime()?;
    let artifact = request
        .vector
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|value| SteeringArtifact::load(Path::new(value)))
        .transpose()?;
    let generated = runtime.generate(
        &request.prompt,
        artifact.as_ref(),
        GenerationOptions {
            strength: request.strength,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
        },
    )?;
    Ok(json!({"text": generated}))
}

fn extract_job(request: ExtractRequest) -> Result<Value> {
    let runtime = request.model.load_runtime()?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    workflow::extract(&runtime, input, output, &layers)?;
    Ok(json!({"path": request.output}))
}

fn inspect_job(request: InspectRequest) -> Result<Value> {
    let artifact = SteeringArtifact::load(Path::new(&request.artifact))
        .with_context(|| format!("failed to inspect {}", request.artifact))?;
    Ok(serde_json::to_value(&artifact)?)
}

// MARK: - Responses

fn send_json(writer: &SharedWriter, status: u16, document: Value) {
    let body = serde_json::to_string_pretty(&document).unwrap_or_default();
    write_response_head(writer, status, "application/json", Some(body.len()));
    write_bytes(writer, body.as_bytes());
}

fn send_error(writer: &SharedWriter, status: u16, message: &str) {
    send_json(writer, status, json!({"error": message}));
}

fn emit_log(writer: &SharedWriter, stream: &str, chunk: &str) {
    emit_event(writer, &json!({"type": "log", "stream": stream, "chunk": chunk}));
}

fn emit_result(writer: &SharedWriter, status: i32, document: Value) {
    emit_event(writer, &json!({"type": "result", "status": status, "json": document}));
}

fn emit_event(writer: &SharedWriter, event: &Value) {
    let mut line = serde_json::to_string(event).unwrap_or_default();
    line.push('\n');
    write_bytes(writer, line.as_bytes());
}

fn write_response_head(
    writer: &SharedWriter,
    status: u16,
    content_type: &str,
    content_length: Option<usize>,
) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let mut head =
        format!("HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\nconnection: close\r\n");
    match content_length {
        Some(length) => head.push_str(&format!("content-length: {length}\r\n\r\n")),
        None => head.push_str("cache-control: no-cache\r\n\r\n"),
    }
    write_bytes(writer, head.as_bytes());
}

fn write_bytes(writer: &SharedWriter, bytes: &[u8]) {
    if let Ok(mut stream) = writer.lock() {
        let _ = stream.write_all(bytes);
        let _ = stream.flush();
    }
}
