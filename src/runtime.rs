use std::{collections::BTreeSet, fs, path::{Path, PathBuf}};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{LlamaConfig, LlamaEosToks},
};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

use crate::{artifact::SteeringArtifact, model::{Cache, SteeringLlama, SteeringPlan}};

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
    pub fn load(model: &str, revision: Option<&str>, device: DeviceChoice) -> Result<Self> {
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
        let builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&source.weights, dtype, &device)
        }.with_context(|| format!("failed to map {} model weight files", source.weights.len()))?;
        let model_impl = SteeringLlama::load(builder, config)?;
        Ok(Self {
            model_id: model.to_owned(),
            revision: source.revision,
            tokenizer,
            model: model_impl,
            device,
            dtype,
            eos_tokens,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.model.config().hidden_size
    }

    pub fn layer_count(&self) -> usize {
        self.model.config().num_hidden_layers
    }

    pub fn activations(&self, prompt: &str, layers: &[usize]) -> Result<Vec<(usize, Vec<f32>)>> {
        validate_layers(layers, self.layer_count())?;
        let ids = self.encode(prompt)?;
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut cache = Cache::new(false, self.dtype, self.model.config(), &self.device)?;
        let output = self.model.forward(&input, 0, &mut cache, None, layers)?;
        Ok(output.activations.into_iter().collect())
    }

    pub fn generate(&self, prompt: &str, artifact: Option<&SteeringArtifact>, options: GenerationOptions) -> Result<String> {
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
            let next = sampler.sample(&output.logits.squeeze(0)?)?;
            tokens.push(next);
            if self.eos_tokens.contains(&next) {
                break;
            }
            if tokens.len() >= self.model.config().max_position_embeddings {
                break;
            }
        }
        self.tokenizer
            .decode(&tokens[prompt_len..], true)
            .map_err(|error| anyhow::anyhow!("failed to decode generated tokens: {error}"))
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
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
