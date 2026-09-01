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
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Context, Result, bail};
use candle_core::Device;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{
    ContrastivePair, DeviceChoice, DpoLoss, DpoOptions, ExampleSet, GenerationOptions, PairSet,
    RewardHead, RewardOptions, Runtime, SftOptions, SteeringArtifact, SynthesisOptions,
    TrainingMethod,
    brama,
    dedupe::DedupeOptions,
    diversity::DEFAULT_MAX_SAMPLE,
    lora,
    pairs::{self, InspectOptions},
    tune,
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
        ("POST", "/v1/pairs/inspect") => stream_job(&writer, &body, job_lock, pairs_inspect_job),
        ("POST", "/v1/pairs/save") => stream_job(&writer, &body, job_lock, pairs_save_job),
        ("POST", "/v1/pairs/synthesize") => {
            stream_job(&writer, &body, job_lock, pairs_synthesize_job)
        }
        ("POST", "/v1/tune/sft") => stream_job(&writer, &body, job_lock, tune_sft_job),
        ("POST", "/v1/tune/dpo") => stream_job(&writer, &body, job_lock, tune_dpo_job),
        ("POST", "/v1/tune/reward") => stream_job(&writer, &body, job_lock, tune_reward_job),
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
    /// A frozen LoRA adapter artifact trained for this exact model. Ster
    /// refuses a mismatch rather than steering the wrong residual stream.
    #[serde(default)]
    adapter: Option<String>,
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairsInspectRequest {
    #[serde(default)]
    pairs: String,
    #[serde(default = "default_dedupe_bits")]
    dedupe_bits: u32,
    #[serde(default = "default_dedupe_bands")]
    dedupe_bands: u32,
    #[serde(default = "default_refusal_threshold")]
    refusal_threshold: f32,
}

impl Validate for PairsInspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.pairs, "pairs inspect requires a pairs file".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairsSaveEntry {
    #[serde(default)]
    positive: String,
    #[serde(default)]
    negative: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairsSaveRequest {
    #[serde(default)]
    path: String,
    #[serde(default)]
    trait_name: String,
    #[serde(default)]
    entries: Vec<PairsSaveEntry>,
}

impl Validate for PairsSaveRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.path, "pairs save requires an output path".to_owned())?;
        if self.entries.is_empty() {
            return Err("pairs save requires at least one pair".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairsSynthesizeRequest {
    /// Which model writes the pairs: local or brama. The model, revision and
    /// device below belong to the local route only; a brama body omits them.
    #[serde(default = "default_generator")]
    generator: String,
    #[serde(default)]
    generator_model: Option<String>,
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default, rename = "trait")]
    trait_description: String,
    #[serde(default)]
    trait_name: String,
    #[serde(default)]
    opposite: Option<String>,
    #[serde(default)]
    count: usize,
    #[serde(default)]
    output: String,
    #[serde(default = "default_retry_multiplier")]
    retry_multiplier: usize,
    #[serde(default = "default_dedupe_bits")]
    dedupe_bits: u32,
    #[serde(default = "default_dedupe_bands")]
    dedupe_bands: u32,
    #[serde(default = "default_refusal_threshold")]
    refusal_threshold: f32,
    #[serde(default = "default_synthesis_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default = "default_synthesis_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    #[serde(default = "default_seed")]
    seed: u64,
}

impl Validate for PairsSynthesizeRequest {
    fn validate(&self) -> Result<(), String> {
        // The local route loads weights and so needs a model; the brama route
        // loads nothing and needs a gateway route instead.
        match self.generator.as_str() {
            "local" => self.model.check("pairs synthesize")?,
            "brama" => require(
                self.generator_model.as_deref().unwrap_or_default(),
                "pairs synthesize with the brama generator requires generatorModel".to_owned(),
            )?,
            _ => return Err("unknown generator; expected local or brama".to_owned()),
        }
        require(
            &self.trait_description,
            "pairs synthesize requires a trait description".to_owned(),
        )?;
        require(&self.output, "pairs synthesize requires an output path".to_owned())?;
        if self.count == 0 {
            return Err("pairs synthesize requires a pair count above zero".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuneSftRequest {
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default)]
    examples: String,
    #[serde(default)]
    output: String,
    #[serde(default = "default_rank")]
    rank: usize,
    #[serde(default = "default_alpha")]
    alpha: f64,
    #[serde(default = "default_targets")]
    targets: String,
    #[serde(default = "default_layers")]
    layers: String,
    #[serde(default = "default_epochs")]
    epochs: usize,
    #[serde(default = "default_learning_rate")]
    learning_rate: f64,
    #[serde(default = "default_accumulation")]
    accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    max_sequence: usize,
    #[serde(default = "default_seed")]
    seed: u64,
}

impl Validate for TuneSftRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune sft")?;
        require(&self.examples, "tune sft requires an example set".to_owned())?;
        require(&self.output, "tune sft requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuneDpoRequest {
    #[serde(flatten)]
    model: ModelRequest,
    /// A contrastive pair set. The positive side is the chosen response and
    /// the negative side the rejected one, so the file Train already reads is
    /// the file this reads.
    #[serde(default)]
    pairs: String,
    #[serde(default)]
    output: String,
    #[serde(default = "default_rank")]
    rank: usize,
    #[serde(default = "default_alpha")]
    alpha: f64,
    #[serde(default = "default_targets")]
    targets: String,
    #[serde(default = "default_layers")]
    layers: String,
    #[serde(default = "default_beta")]
    beta: f64,
    #[serde(default = "default_preference_loss")]
    loss: String,
    #[serde(default = "default_epochs")]
    epochs: usize,
    #[serde(default = "default_learning_rate")]
    learning_rate: f64,
    #[serde(default = "default_accumulation")]
    accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    max_sequence: usize,
    #[serde(default = "default_seed")]
    seed: u64,
}

impl Validate for TuneDpoRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune dpo")?;
        require(&self.pairs, "tune dpo requires a pairs file".to_owned())?;
        require(&self.output, "tune dpo requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuneRewardRequest {
    #[serde(flatten)]
    model: ModelRequest,
    /// A contrastive pair set. The positive side is the response the head
    /// learns to score higher.
    #[serde(default)]
    pairs: String,
    #[serde(default)]
    output: String,
    #[serde(default = "default_rank")]
    rank: usize,
    #[serde(default = "default_alpha")]
    alpha: f64,
    #[serde(default = "default_targets")]
    targets: String,
    #[serde(default = "default_layers")]
    layers: String,
    #[serde(default = "default_epochs")]
    epochs: usize,
    #[serde(default = "default_learning_rate")]
    learning_rate: f64,
    #[serde(default = "default_accumulation")]
    accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    warmup_steps: usize,
    #[serde(default = "default_max_sequence")]
    max_sequence: usize,
    #[serde(default = "default_seed")]
    seed: u64,
}

impl Validate for TuneRewardRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune reward")?;
        require(&self.pairs, "tune reward requires a pairs file".to_owned())?;
        require(&self.output, "tune reward requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuneInspectRequest {
    #[serde(default)]
    artifact: String,
}

impl Validate for TuneInspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.artifact, "tune inspect requires an adapter artifact".to_owned())
    }
}

/// Steering needs a local model, but writing pair text does not, so the
/// hosted route is opt-in and every existing client keeps the local one.
fn default_generator() -> String {
    "local".to_owned()
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

fn default_dedupe_bits() -> u32 {
    3
}

fn default_dedupe_bands() -> u32 {
    8
}

fn default_refusal_threshold() -> f32 {
    0.5
}

fn default_retry_multiplier() -> usize {
    3
}

/// Synthesis answers are one or two sentences, so it keeps a tighter token
/// budget than the general generate endpoint.
fn default_synthesis_max_new_tokens() -> usize {
    96
}

/// Synthesis needs sampling: at zero temperature every attempt would replay
/// the same constant prompt and the run would dedupe to a single pair.
fn default_synthesis_temperature() -> f64 {
    0.9
}

fn default_top_p() -> f64 {
    0.95
}

fn default_rank() -> usize {
    8
}

/// LoRA scales every update by alpha over rank, so sixteen over the default
/// rank of eight is the two-times scale the LoRA papers train with.
fn default_alpha() -> f64 {
    16.0
}

/// Query and value are the projections the LoRA papers adapt first, and the
/// cheapest pair that still moves behaviour.
fn default_targets() -> String {
    "query,value".to_owned()
}

fn default_epochs() -> usize {
    1
}

fn default_learning_rate() -> f64 {
    1e-4
}

fn default_accumulation() -> usize {
    8
}

/// Examples longer than this are skipped rather than truncated; a cut
/// completion would teach the model to stop early.
fn default_max_sequence() -> usize {
    512
}

/// The strength of the pull back toward the frozen reference, at the value the
/// DPO paper reports across its settings.
fn default_beta() -> f64 {
    0.1
}

/// The sigmoid objective the DPO paper derives; `ipo` is the squared-error
/// alternative over length-normalized log-probabilities.
fn default_preference_loss() -> String {
    "dpo".to_owned()
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
    // An adapter rewrites the projections themselves, so it is attached while
    // the weights are mapped rather than applied per token the way a steering
    // vector is.
    let runtime = match request.adapter.as_deref().filter(|value| !value.trim().is_empty()) {
        Some(adapter) => Runtime::load_with_adapter(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            Path::new(adapter),
        )?,
        None => request.model.load_runtime()?,
    };
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

fn pairs_inspect_job(request: PairsInspectRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let options = InspectOptions {
        dedupe: DedupeOptions {
            threshold_bits: request.dedupe_bits,
            num_bands: request.dedupe_bands,
            ..DedupeOptions::default()
        },
        refusal_threshold: request.refusal_threshold,
        ..InspectOptions::default()
    };
    let report = pairs::inspect(&pair_set, &options)?;
    Ok(serde_json::to_value(&report)?)
}

/// The editor's write path. `PairSet::save` validates before it writes, so a
/// set the loader would reject never reaches disk and the desktop sees the
/// same refusal sentence the CLI prints.
fn pairs_save_job(request: PairsSaveRequest) -> Result<Value> {
    let pair_set = PairSet {
        trait_name: request.trait_name,
        pairs: request
            .entries
            .into_iter()
            .map(|entry| ContrastivePair { positive: entry.positive, negative: entry.negative })
            .collect(),
    };
    pair_set.save(Path::new(&request.path))?;
    Ok(json!({"path": request.path, "pairCount": pair_set.pairs.len()}))
}

fn pairs_synthesize_job(request: PairsSynthesizeRequest) -> Result<Value> {
    let options = SynthesisOptions {
        trait_description: request.trait_description,
        trait_name: request.trait_name,
        opposite: request.opposite,
        count: request.count,
        retry_multiplier: request.retry_multiplier,
        dedupe: DedupeOptions {
            threshold_bits: request.dedupe_bits,
            num_bands: request.dedupe_bands,
            ..DedupeOptions::default()
        },
        refusal_threshold: request.refusal_threshold,
        generation: GenerationOptions {
            strength: 1.0,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: Some(request.top_p),
            seed: request.seed,
        },
        diversity_seed: request.seed,
        diversity_max_sample: DEFAULT_MAX_SAMPLE,
    };
    // Same two arms as the CLI, calling the same `pairs::synthesize`: a brama
    // request loads no weights and never touches a device.
    let (pair_set, report) = match request.generator.as_str() {
        "local" => {
            let runtime = request.model.load_runtime()?;
            pairs::synthesize(pairs::Generator::Local(&runtime), &options)?
        }
        "brama" => {
            // `Validate` has already refused a brama request without a route.
            let route = request.generator_model.as_deref().unwrap_or_default();
            let gateway = brama::Gateway::from_env(route)?;
            pairs::synthesize(pairs::Generator::Gateway(&gateway), &options)?
        }
        value => bail!("unknown generator {value:?}; expected local or brama"),
    };
    pair_set.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

/// Mirrors the `ster tune sft` arm: same spec, same `tune::sft`, and the
/// progress lines the trainer writes reach the desktop over the job stream.
fn tune_sft_job(request: TuneSftRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    // The adapters have to exist before the first forward pass, so the
    // runtime is built from the spec rather than patched after loading; the
    // returned VarMap owns every trainable tensor.
    let (runtime, varmap) = Runtime::load_trainable(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
    )?;
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    let options = SftOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::sft(&runtime, &varmap, &examples, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it.
    let artifact = runtime.adapter_artifact(&spec, serde_json::to_value(&report)?)?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

/// Mirrors the `ster tune dpo` arm. One runtime is loaded: the reference the
/// objective measures against is the same weights with the adapters skipped.
fn tune_dpo_job(request: TuneDpoRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    let (runtime, varmap) = Runtime::load_trainable(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
    )?;
    let pairs = PairSet::load(Path::new(&request.pairs))?;
    let options = DpoOptions {
        spec: spec.clone(),
        loss: DpoLoss::parse(&request.loss)?,
        beta: request.beta,
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::dpo(&runtime, &varmap, &pairs, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it.
    let artifact = runtime.adapter_artifact(&spec, serde_json::to_value(&report)?)?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

/// Mirrors the `ster tune reward` arm. The head is registered in the same
/// VarMap the adapters live in, so one optimizer steps the pair and the
/// artifact carries both.
fn tune_reward_job(request: TuneRewardRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    let (runtime, varmap) = Runtime::load_trainable(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
    )?;
    let head =
        RewardHead::fresh(&varmap, runtime.hidden_size(), runtime.device(), runtime.dtype())?;
    let pairs = PairSet::load(Path::new(&request.pairs))?;
    let options = RewardOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::reward(&runtime, &varmap, &head, &pairs, &options)?;
    let artifact =
        runtime.reward_artifact(&spec, head.weight(), serde_json::to_value(&report)?)?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

fn tune_inspect_job(request: TuneInspectRequest) -> Result<Value> {
    // Inspection reads the adapter document alone: no model is loaded, so the
    // tensors land on the CPU whatever trained them.
    let artifact = lora::Artifact::load(Path::new(&request.artifact), &Device::Cpu)
        .with_context(|| format!("failed to inspect {}", request.artifact))?;
    Ok(tune::inspect(&artifact))
}

/// `targets` is a comma-separated projection list, exactly as `--targets` is
/// on the CLI. Repeats collapse and the order follows the request.
fn parse_targets(value: &str) -> Result<Vec<lora::Target>> {
    let mut targets = Vec::new();
    for segment in value.split(',').map(str::trim).filter(|segment| !segment.is_empty()) {
        let target = lora::Target::parse(segment)?;
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    if targets.is_empty() {
        bail!("no targets selected");
    }
    Ok(targets)
}

/// `layers` means what it means everywhere else in Ster, with one difference:
/// `all` cannot be expanded yet. `parse_layers` needs the model's layer count,
/// and the count is only known once the weights are mapped — which happens
/// inside `Runtime::load_trainable`, after the spec exists. An empty layer
/// list is the spec's way of saying every layer, and the loader resolves it
/// against the real count before it builds any adapter.
fn parse_adapter_layers(value: &str) -> Result<Vec<usize>> {
    if value.trim() == "all" {
        return Ok(Vec::new());
    }
    parse_layers(value, usize::MAX)
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
