use std::{collections::{BTreeMap, BTreeSet}, fs, path::{Path, PathBuf}};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{Config, LlamaConfig, LlamaEosToks},
};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

use crate::{
    artifact::SteeringArtifact,
    lora,
    model::{Cache, ForwardOutput, Mode, Route, SteeringLlama, SteeringPlan},
};

pub struct Runtime {
    pub model_id: String,
    pub revision: Option<String>,
    tokenizer: Tokenizer,
    model: SteeringLlama,
    device: Device,
    dtype: DType,
    eos_tokens: BTreeSet<u32>,
}

impl Runtime {
    /// Loads the frozen base model with no adapters attached.
    pub fn load(model: &str, revision: Option<&str>, device: DeviceChoice) -> Result<Self> {
        let base = BaseLoad::resolve(model, revision, device)?;
        let builder = base.builder()?;
        let model_impl = SteeringLlama::load(builder, base.config.clone())?;
        Ok(base.finish(model, model_impl))
    }

    /// Loads a model with frozen LoRA adapters read from an artifact.
    ///
    /// The identity checks mirror the ones `generate` runs against a steering
    /// artifact: an adapter trained against a different checkpoint or a
    /// different width is not merely inaccurate, it is a shape error waiting
    /// to surface halfway through a decode, so it is refused at load time.
    pub fn load_with_adapter(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        adapter: &Path,
    ) -> Result<Self> {
        Ok(Self::load_artifact(model, revision, device, adapter, lora::Kind::Adapter)?.0)
    }

    /// The same load, for an artifact that must be of a stated `kind`, giving
    /// the caller the artifact back as well.
    ///
    /// A reward model's scalar head lives in the same file as its adapters, so
    /// the caller that wants the head needs the document the adapters came out
    /// of — handing it back beats reading the file twice. The kind is required
    /// rather than reported: an artifact that says what it is only helps if
    /// applying it somewhere it does not belong is a refusal.
    pub fn load_artifact(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        path: &Path,
        kind: lora::Kind,
    ) -> Result<(Self, lora::Artifact)> {
        let base = BaseLoad::resolve(model, revision, device)?;
        let artifact = lora::Artifact::load(path, &base.device)?;
        // A reward model's adapters exist to make its head separate, not to
        // change what the model writes; a generation adapter has no head to
        // score with. Either substitution answers a question nobody asked, and
        // does it plausibly, which is the reason it is refused rather than
        // reported.
        match (kind, artifact.kind) {
            (lora::Kind::Adapter, lora::Kind::Reward) => {
                bail!("adapter artifact is a reward model, not a generation adapter")
            }
            (lora::Kind::Reward, lora::Kind::Adapter) => {
                bail!("adapter artifact is a generation adapter, not a reward model")
            }
            _ => {}
        }
        if artifact.model != model {
            bail!(
                "adapter was trained for model {:?}, current model is {:?}",
                artifact.model,
                model
            );
        }
        if artifact.hidden_size != base.config.hidden_size {
            bail!(
                "adapter width {} does not match model width {}",
                artifact.hidden_size,
                base.config.hidden_size
            );
        }
        validate_layers(&artifact.layers, base.config.num_hidden_layers)?;
        let adapters = lora::Adapters::from_artifact(&artifact, &base.device, base.dtype)?;
        let builder = base.builder()?;
        let model_impl = SteeringLlama::load_with_adapters(builder, base.config.clone(), adapters)?;
        Ok((base.finish(model, model_impl), artifact))
    }

    /// Loads a model with fresh trainable adapters; the `VarMap` owns them.
    ///
    /// The base weights are mapped read-only exactly as `load` maps them and
    /// are never registered in the returned map, which is what makes "train
    /// only the adapters" a structural property rather than a convention: the
    /// optimizer is handed `varmap.all_vars()` and there is nothing else in it.
    pub fn load_trainable(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        spec: &lora::Spec,
    ) -> Result<(Self, VarMap)> {
        let base = BaseLoad::resolve(model, revision, device)?;
        spec.validate(base.config.num_hidden_layers)?;
        let spec = spec.resolved(base.config.num_hidden_layers);
        let (hidden, kv_width, intermediate) = projection_widths(&base.config);
        let varmap = VarMap::new();
        let adapters = lora::Adapters::fresh(
            &spec,
            &varmap,
            hidden,
            kv_width,
            intermediate,
            &base.device,
            base.dtype,
        )?;
        let builder = base.builder()?;
        let model_impl = SteeringLlama::load_with_adapters(builder, base.config.clone(), adapters)?;
        Ok((base.finish(model, model_impl), varmap))
    }

    /// The three projection widths an adapter has to match: the residual
    /// width, the width of a fused key or value projection, and the feed
    /// forward width. Grouped-query attention makes the second one smaller
    /// than the first, which is why it cannot be derived from `hidden_size`.
    pub fn config_dims(&self) -> (usize, usize, usize) {
        projection_widths(self.model.config())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Collects this runtime's trained adapters into a durable document.
    ///
    /// `train` is whatever the caller wants recorded about the run that
    /// produced them; `tune::sft` passes its serialized report so the artifact
    /// carries the losses and hyperparameters that made it.
    pub fn adapter_artifact(
        &self,
        spec: &lora::Spec,
        train: serde_json::Value,
    ) -> Result<lora::Artifact> {
        self.artifact(spec, lora::Kind::Adapter, None, train)
    }

    /// The same document plus the scalar head trained on top of these adapters.
    ///
    /// The head goes in the same safetensors file rather than beside it,
    /// because a head and the adapters that shaped the residual stream it
    /// reads are one model: separating them would let an operator pair a head
    /// with adapters it never saw and get scores that mean nothing.
    pub fn reward_artifact(
        &self,
        spec: &lora::Spec,
        head: &Tensor,
        train: serde_json::Value,
    ) -> Result<lora::Artifact> {
        self.artifact(spec, lora::Kind::Reward, Some(head), train)
    }

    fn artifact(
        &self,
        spec: &lora::Spec,
        kind: lora::Kind,
        head: Option<&Tensor>,
        train: serde_json::Value,
    ) -> Result<lora::Artifact> {
        spec.validate(self.layer_count())?;
        let spec = spec.resolved(self.layer_count());
        let adapters = self.model.adapters();
        if adapters.is_empty() {
            bail!("this runtime carries no adapters to write");
        }
        let mut tensors = BTreeMap::new();
        for &layer in &spec.layers {
            for &target in &spec.targets {
                let adapter = adapters.get(layer, target).with_context(|| {
                    format!("layer {layer} carries no {} adapter", target.name())
                })?;
                let (a, b) = lora::Adapter::tensor_names(layer, target);
                tensors.insert(a, adapter.a.clone());
                tensors.insert(b, adapter.b.clone());
            }
        }
        if let Some(head) = head {
            tensors.insert(lora::REWARD_HEAD_TENSOR.to_owned(), head.clone());
        }
        let artifact = lora::Artifact {
            schema_version: lora::ARTIFACT_SCHEMA_VERSION,
            product: "ster".to_owned(),
            kind,
            model: self.model_id.clone(),
            model_revision: self.revision.clone(),
            rank: spec.rank,
            alpha: spec.alpha,
            targets: spec.targets.clone(),
            layers: spec.layers.clone(),
            hidden_size: self.hidden_size(),
            train,
            tensors,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Tokenizes `prompt` and `completion` separately and returns the joined
    /// ids plus the index where the completion begins.
    ///
    /// The split is not cosmetic: it is the only thing that tells the loss
    /// which tokens are the model's answer. Special tokens are added for the
    /// prompt exactly as `encode` adds them, and deliberately not for the
    /// completion — a second begin-of-sequence marker in the middle of the
    /// sequence would be a token the model is asked to predict and never sees
    /// at inference.
    pub fn encode_example(&self, prompt: &str, completion: &str) -> Result<(Vec<u32>, usize)> {
        if completion.trim().is_empty() {
            bail!("training example has an empty completion");
        }
        let mut ids = self.encode(prompt)?;
        let boundary = ids.len();
        let encoded = self
            .tokenizer
            .encode(completion, false)
            .map_err(|error| anyhow::anyhow!("failed to tokenize completion: {error}"))?;
        if encoded.get_ids().is_empty() {
            bail!("training example completion produced no tokens");
        }
        ids.extend_from_slice(encoded.get_ids());
        if ids.len() < 2 {
            bail!("training example encodes to fewer than two tokens, so there is nothing to predict");
        }
        Ok((ids, boundary))
    }

    /// One differentiable forward over `ids`, returning logits `[1, n, vocab]`.
    pub fn forward_train(&self, ids: &[u32]) -> Result<Tensor> {
        self.logits(ids, Mode::TRAIN, "a training forward pass needs at least one token")
    }

    /// One non-differentiable forward over `ids`, returning logits `[1, n, vocab]`.
    ///
    /// `route` picks which model answers. [`Route::Adapted`] is the policy;
    /// [`Route::Base`] is the frozen reference — the same mapped weights with
    /// the low-rank update skipped, which is precisely the model the adapters
    /// started as, since `B` is zeros before the first step. Preference
    /// optimization gets its reference log-probabilities this way rather than
    /// by loading a second copy of the checkpoint.
    ///
    /// Nothing here is backpropagated, so it takes the fused kernels. A caller
    /// holding a trainable runtime must not route a *policy* score through it:
    /// the adapter variables would record a graph whose rope, softmax and
    /// norm nodes have no backward pass. That caller wants `forward_train`.
    pub fn forward_scored(&self, ids: &[u32], route: Route) -> Result<Tensor> {
        self.logits(ids, Mode::score(route), "a scoring forward pass needs at least one token")
    }

    /// One differentiable forward over `ids`, returning the residual stream
    /// after the final norm, `[1, n, hidden]`, and no vocabulary projection.
    ///
    /// This is what a reward head reads. Skipping the vocabulary matmul is not
    /// a micro-optimization on a real checkpoint: it is the widest matmul in
    /// the pass, and a reward run would compute and backpropagate all of it
    /// only to throw the result away.
    pub fn forward_hidden(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .forward_once(ids, Mode::REWARD, "a reward forward pass needs at least one token")?
            .hidden)
    }

    /// The same residual stream with no autograd tape, for a reward model that
    /// is judging rather than being trained.
    ///
    /// A reward model inside a policy-optimization loop is frozen by
    /// definition — a moving judge is a moving target — so it takes the fused
    /// kernels and records nothing.
    pub fn forward_hidden_scored(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self
            .forward_once(ids, Mode::JUDGE, "a scoring forward pass needs at least one token")?
            .hidden)
    }

    /// A forward that must produce logits, unwrapped.
    fn logits(&self, ids: &[u32], mode: Mode, empty: &str) -> Result<Tensor> {
        self.forward_once(ids, mode, empty)?
            .logits
            .context("this forward pass was asked for no vocabulary projection")
    }

    /// The body every whole-sequence forward shares: one sequence, no KV cache.
    ///
    /// The cache stays off because the whole sequence goes through in one pass,
    /// so there is nothing to reuse, and because a cache would keep the previous
    /// sequence's keys and values alive inside this one's autograd graph — the
    /// backward pass would then walk tensors that no longer correspond to the
    /// input being scored.
    fn forward_once(&self, ids: &[u32], mode: Mode, empty: &str) -> Result<ForwardOutput> {
        if ids.is_empty() {
            bail!("{empty}");
        }
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mut cache = Cache::new(false, self.dtype, self.model.config(), &self.device)?;
        Ok(self.model.forward_pass(&input, 0, &mut cache, None, &[], mode)?)
    }

    pub fn hidden_size(&self) -> usize {
        self.model.config().hidden_size
    }

    pub fn layer_count(&self) -> usize {
        self.model.config().num_hidden_layers
    }

    /// The longest sequence the rotary tables and the position mask cover.
    /// Training clamps against it for the same reason `generate` does: past
    /// it the cached angles simply do not exist.
    pub fn context_length(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    pub fn activations(&self, prompt: &str, layers: &[usize]) -> Result<Vec<(usize, Vec<f32>)>> {
        validate_layers(layers, self.layer_count())?;
        let ids = self.encode(prompt)?;
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut cache = Cache::new(false, self.dtype, self.model.config(), &self.device)?;
        let output = self.model.forward(&input, 0, &mut cache, None, layers)?;
        Ok(output.activations.into_iter().collect())
    }

    /// One sampled continuation, decoded.
    ///
    /// Everything here is [`Runtime::sample`]; only the text survives, which
    /// is what every caller outside policy optimization wants.
    pub fn generate(
        &self,
        prompt: &str,
        artifact: Option<&SteeringArtifact>,
        options: GenerationOptions,
    ) -> Result<String> {
        Ok(self.sample(prompt, artifact, options)?.text)
    }

    /// One sampled continuation: the prompt as the sampler tokenized it, the
    /// tokens drawn after it, and their text.
    ///
    /// Policy optimization needs all three. It has to score the exact sequence
    /// the policy produced, and decoding to text and re-encoding would not
    /// reliably give that sequence back — a tokenizer is not injective over
    /// its own output. Handing back the ids the sampler actually pushed makes
    /// the scored sequence the sampled sequence by construction.
    pub fn sample(
        &self,
        prompt: &str,
        artifact: Option<&SteeringArtifact>,
        options: GenerationOptions,
    ) -> Result<Completion> {
        if options.max_new_tokens == 0 {
            bail!("max_new_tokens must be greater than zero");
        }
        let mut tokens = self.encode(prompt)?;
        if tokens.len() >= self.model.config().max_position_embeddings {
            bail!(
                "prompt contains {} tokens, model context allows fewer than {}",
                tokens.len(),
                self.model.config().max_position_embeddings
            );
        }
        let prompt_len = tokens.len();
        let plan = match artifact {
            Some(artifact) => {
                artifact.validate()?;
                if artifact.model != self.model_id {
                    bail!(
                        "artifact was trained for model {:?}, current model is {:?}",
                        artifact.model,
                        self.model_id
                    );
                }
                if artifact.hidden_size != self.hidden_size() {
                    bail!(
                        "artifact width {} does not match model width {}",
                        artifact.hidden_size,
                        self.hidden_size()
                    );
                }
                validate_layers(
                    &artifact.vectors.iter().map(|vector| vector.layer).collect::<Vec<_>>(),
                    self.layer_count(),
                )?;
                Some(SteeringPlan::new(
                    artifact.vectors.iter().map(|vector| (vector.layer, vector.values.clone())),
                    options.strength,
                    self.hidden_size(),
                    &self.device,
                    self.dtype,
                )?)
            }
            None => None,
        };
        let sampling = if options.temperature <= 0.0 {
            Sampling::ArgMax
        } else if let Some(top_p) = options.top_p {
            Sampling::TopP { p: top_p, temperature: options.temperature }
        } else {
            Sampling::All { temperature: options.temperature }
        };
        let mut sampler = LogitsProcessor::from_sampling(options.seed, sampling);
        let mut cache = Cache::new(true, self.dtype, self.model.config(), &self.device)?;
        for step in 0..options.max_new_tokens {
            let (context, index_pos) = if step == 0 {
                (tokens.clone(), 0)
            } else {
                (vec![*tokens.last().expect("tokens are non-empty")], tokens.len() - 1)
            };
            let input = Tensor::new(context.as_slice(), &self.device)?.unsqueeze(0)?;
            let output = self.model.forward(&input, index_pos, &mut cache, plan.as_ref(), &[])?;
            // `Mode::DECODE` always asks for the last position's logits, so
            // this is the one readout that cannot be absent.
            let logits = output.logits.context("the decode pass produced no logits")?;
            let next = sampler.sample(&logits.squeeze(0)?)?;
            tokens.push(next);
            if self.eos_tokens.contains(&next) {
                break;
            }
            if tokens.len() >= self.model.config().max_position_embeddings {
                break;
            }
        }
        let text = self
            .tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(|error| anyhow::anyhow!("failed to decode generated tokens: {error}"))?;
        let sampled = tokens.split_off(prompt_len);
        Ok(Completion { prompt: tokens, tokens: sampled, text })
    }

    /// Tokenizes one text with the tokenizer's own special tokens, exactly as
    /// every prompt in Ster is tokenized.
    ///
    /// Preference optimization scores whole texts rather than prompt and
    /// completion halves, so it needs this rather than `encode_example`; the
    /// begin-of-sequence marker the tokenizer prepends is what gives the first
    /// real token a position to be predicted from.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        if prompt.trim().is_empty() {
            bail!("prompt must not be empty");
        }
        let encoded = self.tokenizer
            .encode(prompt, true)
            .map_err(|error| anyhow::anyhow!("failed to tokenize prompt: {error}"))?;
        let ids = encoded.get_ids().to_vec();
        if ids.is_empty() {
            bail!("tokenizer produced no tokens");
        }
        Ok(ids)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceChoice {
    Cpu,
    Metal,
    Cuda,
}

impl DeviceChoice {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => bail!("unknown device {value:?}; expected cpu, metal, or cuda"),
        }
    }

    fn resolve(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Metal => {
                #[cfg(feature = "metal")]
                { Device::new_metal(0).context("failed to initialize Metal device") }
                #[cfg(not(feature = "metal"))]
                { bail!("this Ster binary was built without the metal feature") }
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                { Device::new_cuda(0).context("failed to initialize CUDA device") }
                #[cfg(not(feature = "cuda"))]
                { bail!("this Ster binary was built without the cuda feature") }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationOptions {
    pub strength: f64,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
}

/// What one sampling call produced.
///
/// The two token vectors concatenate to the exact sequence the model saw, and
/// `prompt.len()` is the boundary a completion-only loss scores from — which
/// is why the prompt travels back out rather than being re-derived: a caller
/// that tokenized the prompt itself would be trusting two encodes to agree.
#[derive(Debug, Clone)]
pub struct Completion {
    pub prompt: Vec<u32>,
    pub tokens: Vec<u32>,
    pub text: String,
}

/// Everything the three loaders share, held between resolving the checkpoint
/// and mapping its weights.
///
/// The split exists because `load_trainable` has to size its adapters from the
/// config *before* the base weights are mapped, and because three copies of
/// the config parsing, the architecture refusal, and the tokenizer load would
/// drift the moment one of them gained a check the other two did not.
struct BaseLoad {
    tokenizer: Tokenizer,
    config: Config,
    weights: Vec<PathBuf>,
    revision: Option<String>,
    eos_tokens: BTreeSet<u32>,
    device: Device,
    dtype: DType,
}

impl BaseLoad {
    fn resolve(model: &str, revision: Option<&str>, device: DeviceChoice) -> Result<Self> {
        let device = device.resolve()?;
        let dtype = DType::F32;
        let source = resolve_model(model, revision)?;
        let config_bytes = fs::read(&source.config)
            .with_context(|| format!("failed to read {}", source.config.display()))?;
        let raw_config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("invalid model config {}", source.config.display()))?;
        let model_type = raw_config.get("model_type").and_then(|value| value.as_str()).unwrap_or("");
        if model_type != "llama" {
            bail!(
                "model architecture {model_type:?} is unsupported by this Ster build; use a Hugging Face Llama-family checkpoint with model_type=llama"
            );
        }
        let llama: LlamaConfig = serde_json::from_slice(&config_bytes)
            .with_context(|| format!("invalid Llama config {}", source.config.display()))?;
        let eos_tokens = eos_tokens(&llama);
        let config = llama.into_config(false);
        let tokenizer = Tokenizer::from_file(&source.tokenizer)
            .map_err(|error| anyhow::anyhow!("failed to load tokenizer {}: {error}", source.tokenizer.display()))?;
        Ok(Self {
            tokenizer,
            config,
            weights: source.weights,
            revision: source.revision,
            eos_tokens,
            device,
            dtype,
        })
    }

    /// Maps the base weights read-only. Nothing here is registered in a
    /// `VarMap`, so the base stays frozen whichever loader called it.
    fn builder(&self) -> Result<VarBuilder<'static>> {
        unsafe { VarBuilder::from_mmaped_safetensors(&self.weights, self.dtype, &self.device) }
            .with_context(|| format!("failed to map {} model weight files", self.weights.len()))
    }

    fn finish(self, model_id: &str, model: SteeringLlama) -> Runtime {
        Runtime {
            model_id: model_id.to_owned(),
            revision: self.revision,
            tokenizer: self.tokenizer,
            model,
            device: self.device,
            dtype: self.dtype,
            eos_tokens: self.eos_tokens,
        }
    }
}

/// Residual width, fused key/value projection width, feed forward width.
///
/// Grouped-query attention gives the key and value projections fewer heads
/// than the query projection, so their output is `num_key_value_heads *
/// head_dim` wide rather than `hidden_size` wide. An adapter sized from
/// `hidden_size` would fail to matmul against them.
fn projection_widths(config: &Config) -> (usize, usize, usize) {
    let head_dim = config.hidden_size / config.num_attention_heads;
    (
        config.hidden_size,
        config.num_key_value_heads * head_dim,
        config.intermediate_size,
    )
}

struct ResolvedModel {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: Vec<PathBuf>,
    revision: Option<String>,
}

fn resolve_model(model: &str, revision: Option<&str>) -> Result<ResolvedModel> {
    let local = Path::new(model);
    if local.is_dir() {
        let config = local.join("config.json");
        let tokenizer = local.join("tokenizer.json");
        let weights = local_safetensors(local)?;
        require_files(&config, &tokenizer, &weights)?;
        return Ok(ResolvedModel { config, tokenizer, weights, revision: revision.map(str::to_owned) });
    }
    let api = Api::new().context("failed to initialize Hugging Face Hub client")?;
    let repo = Repo::with_revision(
        model.to_owned(),
        RepoType::Model,
        revision.unwrap_or("main").to_owned(),
    );
    let remote = api.repo(repo);
    let info = remote.info().with_context(|| format!("failed to read model repository {model}"))?;
    let config = remote.get("config.json")?;
    let tokenizer = remote.get("tokenizer.json")?;
    let weight_names: Vec<String> = info.siblings
        .into_iter()
        .map(|file| file.rfilename)
        .filter(|name| {
            name.ends_with(".safetensors")
                && !name.contains("optimizer")
                && !name.contains("training_args")
        })
        .collect();
    if weight_names.is_empty() {
        bail!("model {model} publishes no safetensors weights");
    }
    let mut weights = Vec::with_capacity(weight_names.len());
    for name in weight_names {
        weights.push(remote.get(&name).with_context(|| format!("failed to download {name}"))?);
    }
    require_files(&config, &tokenizer, &weights)?;
    Ok(ResolvedModel { config, tokenizer, weights, revision: Some(info.sha) })
}

fn local_safetensors(root: &Path) -> Result<Vec<PathBuf>> {
    let mut weights = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("failed to list {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
            && !path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.contains("optimizer"))
        {
            weights.push(path);
        }
    }
    weights.sort();
    Ok(weights)
}

fn require_files(config: &Path, tokenizer: &Path, weights: &[PathBuf]) -> Result<()> {
    if !config.is_file() {
        bail!("model config is missing: {}", config.display());
    }
    if !tokenizer.is_file() {
        bail!("tokenizer is missing: {}", tokenizer.display());
    }
    if weights.is_empty() {
        bail!("model directory contains no safetensors weights");
    }
    Ok(())
}

fn validate_layers(layers: &[usize], count: usize) -> Result<()> {
    if layers.is_empty() {
        bail!("at least one layer is required");
    }
    if let Some(layer) = layers.iter().copied().find(|layer| *layer >= count) {
        bail!("layer {layer} is outside the model's 0..{} range", count.saturating_sub(1));
    }
    Ok(())
}

fn eos_tokens(config: &LlamaConfig) -> BTreeSet<u32> {
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => [*token].into_iter().collect(),
        Some(LlamaEosToks::Multiple(tokens)) => tokens.iter().copied().collect(),
        None => BTreeSet::new(),
    }
}
