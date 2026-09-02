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
    ChatChoice, ContrastivePair, DeviceChoice, DpoLoss, DpoOptions, EvaluateOptions, ExampleSet, GenerationOptions, GrpoOptions,
    PairSet, Precision, PromptSet, Reward, RewardHead, RewardOptions, Runtime, SftOptions, SteeringArtifact,
    SynthesisOptions, TrainingMethod,
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
    /// `auto` renders the prompt through the model's own chat template when
    /// it publishes one, `off` sends raw text. An instruct checkpoint handed
    /// a bare prompt continues it instead of answering it.
    #[serde(default = "default_chat_template")]
    chat_template: String,
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
    #[serde(default = "default_chat_template")]
    chat_template: String,
    /// Rows folded into one forward pass — examples here, pairs on the
    /// preference endpoints, where a pair is two rows. One is the unbatched
    /// pass every run recorded so far took.
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    /// The dtype the frozen base weights are mapped at: `f32`, `f16`, or
    /// `bf16`. Adapters, any head, and every optimizer moment stay in f32
    /// whatever this says. `bf16` needs the `metal` device.
    #[serde(default = "default_precision")]
    precision: String,
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
    #[serde(default = "default_chat_template")]
    chat_template: String,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_precision")]
    precision: String,
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
    #[serde(default = "default_chat_template")]
    chat_template: String,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_precision")]
    precision: String,
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
struct TuneGrpoRequest {
    #[serde(flatten)]
    model: ModelRequest,
    /// A prompt set, `{"prompts": ["…"]}` — the shape extract already takes.
    #[serde(default)]
    prompts: String,
    #[serde(default)]
    output: String,
    /// The keyword `length`, or the path to a reward artifact.
    #[serde(default = "default_reward")]
    reward: String,
    #[serde(default = "default_group")]
    group: usize,
    #[serde(default = "default_iterations")]
    iterations: usize,
    #[serde(default = "default_kl_beta")]
    beta: f64,
    #[serde(default = "default_rank")]
    rank: usize,
    #[serde(default = "default_alpha")]
    alpha: f64,
    #[serde(default = "default_targets")]
    targets: String,
    #[serde(default = "default_layers")]
    layers: String,
    #[serde(default = "default_learning_rate")]
    learning_rate: f64,
    /// One group is already `group` sequences, so a step per group is the
    /// natural unit and the default is one rather than eight.
    #[serde(default = "default_group_accumulation")]
    accumulation: usize,
    /// Zero starts at the full learning rate, which is what a short run wants.
    #[serde(default)]
    warmup_steps: usize,
    #[serde(default = "default_grpo_max_new_tokens")]
    max_new_tokens: usize,
    #[serde(default = "default_grpo_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    #[serde(default = "default_max_sequence")]
    max_sequence: usize,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_chat_template")]
    chat_template: String,
    #[serde(default = "default_precision")]
    precision: String,
}

impl Validate for TuneGrpoRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune grpo")?;
        require(&self.prompts, "tune grpo requires a prompt set".to_owned())?;
        require(&self.output, "tune grpo requires an output path".to_owned())?;
        require(&self.reward, "tune grpo requires a reward source".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuneMergeRequest {
    #[serde(flatten)]
    model: ModelRequest,
    /// The adapter to fold in; it must be a generation adapter trained for
    /// this exact model.
    #[serde(default)]
    adapter: String,
    /// Directory to write: model.safetensors plus the source's own
    /// config.json and tokenizer.json, which is what `model` accepts.
    #[serde(default)]
    output: String,
}

impl Validate for TuneMergeRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune merge")?;
        require(&self.adapter, "tune merge requires an adapter".to_owned())?;
        require(&self.output, "tune merge requires an output directory".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TuneEvaluateRequest {
    #[serde(flatten)]
    model: ModelRequest,
    #[serde(default)]
    examples: String,
    /// A frozen adapter to attach before scoring; omit or leave empty to score
    /// the bare checkpoint, which is the run an adapter is compared against.
    #[serde(default)]
    adapter: Option<String>,
    #[serde(default = "default_max_sequence")]
    max_sequence: usize,
    #[serde(default = "default_chat_template")]
    chat_template: String,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_precision")]
    precision: String,
}

impl Validate for TuneEvaluateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune evaluate")?;
        require(&self.examples, "tune evaluate requires an example set".to_owned())
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

/// Apply the model's own conversation format when it publishes one. An
/// instruct checkpoint is the common case and raw text is wrong for it, so
/// the default is the setting that is right more often; `off` restores the
/// raw-text encoding a base model wants.
fn default_chat_template() -> String {
    "auto".to_owned()
}

/// One sequence per forward: the unbatched pass every run recorded before
/// batching existed, so a client that does not ask keeps its numbers.
fn default_batch_size() -> usize {
    1
}

/// Single precision, which is what every recorded run used. Half precision is
/// opt-in because it changes the numbers a client may be comparing against.
fn default_precision() -> String {
    "f32".to_owned()
}

/// Records the dtype the base weights were mapped at in a run's own report,
/// beside the chat-template decision and for the same reason: two runs of the
/// same request at different precisions produce different losses, and a report
/// that does not say which one made it is not comparable with the other.
fn note_precision(report: &mut Value, precision: Precision) -> Result<()> {
    report
        .as_object_mut()
        .context("a run report must be a JSON object to record its precision")?
        .insert("precision".to_owned(), json!(precision.name()));
    Ok(())
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

/// The offline reward: a completion's sampled-token count, which needs no
/// judge and no artifact, so the loop is runnable the first time it is asked
/// for.
fn default_reward() -> String {
    "length".to_owned()
}

/// Four completions per prompt: enough for a baseline that is not the sample
/// itself, cheap enough that a first run finishes.
fn default_group() -> usize {
    4
}

fn default_iterations() -> usize {
    1
}

/// The KL weight the GRPO paper reports; smaller than a preference loss's beta
/// because it is a penalty per token rather than a scale on the whole margin.
fn default_kl_beta() -> f64 {
    0.04
}

/// One prompt group is already `group` sequences, so a step per group is the
/// natural unit and this default is one where the other trainers use eight.
fn default_group_accumulation() -> usize {
    1
}

fn default_grpo_max_new_tokens() -> usize {
    64
}

/// Policy optimization needs sampling: at zero temperature every draw in a
/// group would be the same completion and the baseline would have nothing to
/// compare against.
fn default_grpo_temperature() -> f64 {
    0.9
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
    let mut runtime = match request.adapter.as_deref().filter(|value| !value.trim().is_empty()) {
        Some(adapter) => Runtime::load_with_adapter(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            Path::new(adapter),
        )?,
        None => request.model.load_runtime()?,
    };
    runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
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
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    let options = SftOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::sft(&runtime, &varmap, &examples, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it, and the encoding it was produced in
    // travels with it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
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
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let pairs = PairSet::load(Path::new(&request.pairs))?;
    let options = DpoOptions {
        spec: spec.clone(),
        loss: DpoLoss::parse(&request.loss)?,
        beta: request.beta,
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::dpo(&runtime, &varmap, &pairs, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
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
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    // The head is registered at the parameter dtype, never the base dtype: a
    // scalar head is exactly the small trained weight that rounds away in
    // half precision.
    let head =
        RewardHead::fresh(&varmap, runtime.hidden_size(), runtime.device(), runtime.param_dtype())?;
    let pairs = PairSet::load(Path::new(&request.pairs))?;
    let options = RewardOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::reward(&runtime, &varmap, &head, &pairs, &options)?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.reward_artifact(&spec, head.weight(), report.clone())?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

/// Mirrors the `ster tune grpo` arm. The reward source is resolved before the
/// policy is loaded, so a reward artifact for the wrong checkpoint is refused
/// before the desktop waits out a policy load to hear it.
fn tune_grpo_job(request: TuneGrpoRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    let source = Reward::parse(
        &request.reward,
        &request.model.model,
        request.model.revision.as_deref(),
        device,
    )?;
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let prompts = PromptSet::load(Path::new(&request.prompts))?;
    let options = GrpoOptions {
        spec: spec.clone(),
        group: request.group,
        iterations: request.iterations,
        beta: request.beta,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        generation: GenerationOptions {
            strength: 1.0,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: Some(request.top_p),
            seed: request.seed,
        },
    };
    let report = tune::grpo(&runtime, &varmap, &prompts, &source, &request.reward, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

/// Mirrors the `ster tune merge` arm. No device and no runtime: merging
/// rewrites tensors and never runs the model.
fn tune_merge_job(request: TuneMergeRequest) -> Result<Value> {
    let report = tune::merge(
        &request.model.model,
        request.model.revision.as_deref(),
        Path::new(&request.adapter),
        Path::new(&request.output),
    )?;
    Ok(json!({ "report": report }))
}

/// Mirrors the `ster tune evaluate` arm: no optimizer, no artifact written,
/// and the same document the CLI prints.
fn tune_evaluate_job(request: TuneEvaluateRequest) -> Result<Value> {
    let adapter = request.adapter.as_deref().filter(|value| !value.trim().is_empty());
    let precision = Precision::parse(&request.precision)?;
    let mut runtime = match adapter {
        Some(adapter) => Runtime::load_with_adapter_at(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            Path::new(adapter),
            precision,
        )?,
        None => Runtime::load_at(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            precision,
        )?,
    };
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    let report = tune::evaluate(
        &runtime,
        &examples,
        adapter.map(Path::new),
        &EvaluateOptions { max_sequence: request.max_sequence, batch: request.batch_size },
    )?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    Ok(report)
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
