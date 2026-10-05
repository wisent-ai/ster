//! Which files a checkpoint is, resolved once from a local directory or the
//! hub, and the config questions every loader asks of them.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::{Config, LlamaConfig, LlamaEosToks};
use hf_hub::{Repo, RepoType, api::sync::Api};

use crate::{chat, model::Architecture};

use super::{
    config,
    family::{Family, family, fill_llama_keys, take_rope_scaling},
    mistral::{self, Layout},
};

/// A checkpoint's three files, resolved but not mapped.
///
/// `Runtime::load` maps the weights the moment it resolves them, which is what
/// every command that runs the model wants and exactly what merging one does
/// not: folding an adapter into the base rewrites tensors and never builds a
/// decoder. Splitting resolution from mapping is what lets it read the same
/// files, through the same Hub path and the same architecture refusal, without
/// paying for a model it will not run.
pub struct Checkpoint {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
    pub revision: Option<String>,
    /// The tokenizer's own configuration, which is where a checkpoint records
    /// its chat template and the text of its special tokens, and the
    /// standalone template file newer repositories use instead. Both are
    /// optional: a base model publishes neither, and that is a fact to report
    /// rather than a checkpoint to refuse.
    pub tokenizer_config: Option<PathBuf>,
    pub chat_template: Option<PathBuf>,
    /// Whether the config and weights are Transformers' or Mistral's own.
    pub layout: Layout,
}

impl Checkpoint {
    /// Resolves `model` from a local directory or the Hugging Face Hub.
    pub fn resolve(model: &str, revision: Option<&str>) -> Result<Self> {
        let local = Path::new(model);
        if local.is_dir() {
            let layout = if local.join("config.json").is_file() || !local.join("params.json").is_file() {
                Layout::Transformers
            } else {
                Layout::Mistral
            };
            let config = local.join(layout.config_file());
            let tokenizer = local.join(tokenizer_file(|name| local.join(name).is_file()));
            let weights = local_safetensors(local, layout)?;
            require_files(&config, &tokenizer, &weights)?;
            let (tokenizer_config, chat_template) = chat::local_files(local);
            return Ok(Self {
                config,
                tokenizer,
                weights,
                revision: revision.map(str::to_owned),
                tokenizer_config,
                chat_template,
                layout,
            });
        }
        let api = Api::new().context("failed to initialize Hugging Face Hub client")?;
        let repo = Repo::with_revision(
            model.to_owned(),
            RepoType::Model,
            revision.unwrap_or("main").to_owned(),
        );
        let remote = api.repo(repo);
        let info = remote
            .info()
            .with_context(|| format!("failed to read model repository {model}"))?;
        let tokenizer = remote.get(tokenizer_file(|name| published(&info, name)))?;
        // The two template files are fetched exactly like the three required
        // ones, but only when the repository lists them: `get` on a file a
        // repository does not publish is an error, and a base model not
        // publishing a chat template is not an error.
        let has_tokenizer_config = published(&info, "tokenizer_config.json");
        let has_chat_template = published(&info, "chat_template.jinja");
        let tokenizer_config = has_tokenizer_config
            .then(|| remote.get("tokenizer_config.json"))
            .transpose()
            .context("failed to download tokenizer_config.json")?;
        let chat_template = has_chat_template
            .then(|| remote.get("chat_template.jinja"))
            .transpose()
            .context("failed to download chat_template.jinja")?;
        // A repository that publishes Mistral's own format alone (Mistral
        // Large 3) is read from `params.json` and its consolidated weights.
        let layout = if published(&info, "config.json") || !published(&info, "params.json") {
            Layout::Transformers
        } else {
            Layout::Mistral
        };
        let config = remote.get(layout.config_file())?;
        // The architecture and quantization refusals come before the
        // weights: a checkpoint Ster cannot run is refused for the price of
        // its config, not of every shard.
        supported(&config::read(&config, layout)?.0, &config)?;
        let weight_names: Vec<String> = info
            .siblings
            .into_iter()
            .map(|file| file.rfilename)
            .filter(|name| layout.owns(name))
            .collect();
        if weight_names.is_empty() {
            bail!("model {model} publishes no safetensors weights");
        }
        let mut weights = Vec::with_capacity(weight_names.len());
        for name in weight_names {
            weights.push(
                remote
                    .get(&name)
                    .with_context(|| format!("failed to download {name}"))?,
            );
        }
        require_files(&config, &tokenizer, &weights)?;
        Ok(Self {
            config,
            tokenizer,
            weights,
            revision: Some(info.sha),
            tokenizer_config,
            chat_template,
            layout,
        })
    }

    /// The parsed decoder config, what the architecture adds to it, and its
    /// end-of-sequence tokens.
    ///
    /// The architecture refusal lives here rather than in each caller, so a
    /// command that never builds a decoder still refuses a checkpoint the
    /// decoder could not have loaded — a merge that silently produced a
    /// directory `Runtime::load` then rejects would be worse than no merge.
    pub fn decoder_config(&self) -> Result<(Config, Architecture, BTreeSet<u32>)> {
        let (mut raw, wrapper) = config::read(&self.config, self.layout)?;
        // Mistral's own format leaves the end-of-sequence token to its
        // tokenizer.
        if let (Layout::Mistral, Some(object)) = (self.layout, raw.as_object_mut()) {
            if let Some(eos) = mistral::end_of_sequence(self.tokenizer_config.as_deref(), &self.tokenizer) {
                object.entry("eos_token_id").or_insert(serde_json::Value::from(eos));
            }
        }
        let (model_type, found) = supported(&raw, &self.config)?;
        let model_type = model_type.as_str();
        let scaling = take_rope_scaling(&mut raw, &self.config)?;
        fill_llama_keys(&mut raw, model_type);
        let mut llama: LlamaConfig = serde_json::from_value(raw.clone())
            .with_context(|| format!("invalid {model_type} config {}", self.config.display()))?;
        if found.tied_by_default() && llama.tie_word_embeddings.is_none() {
            // These families tie their word embeddings by default and their
            // configs often leave the key out.
            llama.tie_word_embeddings = Some(true);
        }
        let mut architecture = family(model_type, &raw, scaling.as_ref(), &llama, &self.config)?;
        architecture.names.wrapper = wrapper;
        // A looped model's layer count is every pass's layers, so the cache,
        // steering and capture address each pass's layers apart.
        if let Some(loops) = architecture.loops {
            llama.num_hidden_layers = loops.physical * loops.count;
        }
        if let Some(recurrence) = architecture.recurrence {
            llama.num_hidden_layers = recurrence.layers();
        }
        let tokens = eos_tokens(&llama);
        Ok((llama.into_config(false), architecture, tokens))
    }

    /// The conversation format this checkpoint publishes, if it publishes one.
    ///
    /// Compiled here rather than at first use so a template that does not
    /// parse is refused while the operator is still waiting on the load,
    /// instead of halfway through an epoch.
    pub fn chat(&self) -> Result<Option<chat::Template>> {
        chat::Template::load(
            self.tokenizer_config.as_deref(),
            self.chat_template.as_deref(),
        )
    }
}

/// The family `raw` (a decoder config read by [`config::read`]) belongs to,
/// and its `model_type`, or the refusal of a model Ster does not implement
/// or of a quantized checkpoint. A remote-code config may leave `model_type`
/// to its config class (LongCat-Flash-Lite); its `architectures` class then
/// names it.
fn supported(raw: &serde_json::Value, path: &Path) -> Result<(String, Family)> {
    let model_type = match raw.get("model_type").and_then(|value| value.as_str()) {
        Some(stated) => stated.to_owned(),
        None => Family::of_architectures(raw).map(Family::model_type).unwrap_or("").to_owned(),
    };
    let Some(found) = Family::of(&model_type) else {
        let families: Vec<&str> = Family::ALL.iter().map(|family| family.model_type()).collect();
        bail!(
            "model architecture {model_type:?} is unsupported by this Ster build; use a Hugging Face checkpoint whose model_type is one of {}",
            families.join(", ")
        );
    };
    if raw.get("quantization_config").is_some() {
        bail!(
            "{} is a quantized checkpoint (it declares quantization_config); Ster maps unquantized safetensors only, so use the checkpoint it was quantized from",
            path.display()
        );
    }
    Ok((model_type, found))
}

/// Whether the repository lists a file, so an optional one is only fetched
/// when asking for it can succeed.
fn published(info: &hf_hub::api::RepoInfo, name: &str) -> bool {
    info.siblings.iter().any(|file| file.rfilename == name)
}

/// The tokenizer file a checkpoint publishes: Transformers'
/// `tokenizer.json`, else PLaMo's `tokenizer.jsonl`, else a
/// `tokenizer.model` (GLM-4's tiktoken vocabulary), else Kimi's
/// `tiktoken.model`, else the `tokenizer.json` whose absence is then
/// reported.
fn tokenizer_file(exists: impl Fn(&str) -> bool) -> &'static str {
    ["tokenizer.json", "tokenizer.jsonl", "tokenizer.model", "tiktoken.model"]
        .into_iter()
        .find(|name| exists(name))
        .unwrap_or("tokenizer.json")
}

fn local_safetensors(root: &Path, layout: Layout) -> Result<Vec<PathBuf>> {
    let mut weights = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("failed to list {}", root.display()))? {
        let path = entry?.path();
        if path.file_name().and_then(|name| name.to_str()).is_some_and(|name| layout.owns(name)) {
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

fn eos_tokens(config: &LlamaConfig) -> BTreeSet<u32> {
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => [*token].into_iter().collect(),
        Some(LlamaEosToks::Multiple(tokens)) => tokens.iter().copied().collect(),
        None => BTreeSet::new(),
    }
}
