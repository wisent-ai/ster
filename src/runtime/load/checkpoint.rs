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

use super::family::{Family, family, fill_llama_keys, take_rope_scaling};

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
}

impl Checkpoint {
    /// Resolves `model` from a local directory or the Hugging Face Hub.
    pub fn resolve(model: &str, revision: Option<&str>) -> Result<Self> {
        let local = Path::new(model);
        if local.is_dir() {
            let config = local.join("config.json");
            let tokenizer = local.join("tokenizer.json");
            let weights = local_safetensors(local)?;
            require_files(&config, &tokenizer, &weights)?;
            let (tokenizer_config, chat_template) = chat::local_files(local);
            return Ok(Self {
                config,
                tokenizer,
                weights,
                revision: revision.map(str::to_owned),
                tokenizer_config,
                chat_template,
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
        let config = remote.get("config.json")?;
        let tokenizer = remote.get("tokenizer.json")?;
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
        let weight_names: Vec<String> = info
            .siblings
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
        let bytes = fs::read(&self.config)
            .with_context(|| format!("failed to read {}", self.config.display()))?;
        let outer: serde_json::Value = serde_json::from_slice(&finite_literals(&bytes))
            .with_context(|| format!("invalid model config {}", self.config.display()))?;
        let (mut raw, wrapper) = language_model(outer);
        let model_type = raw
            .get("model_type")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_owned();
        let model_type = model_type.as_str();
        let Some(found) = Family::of(model_type) else {
            let families: Vec<&str> = Family::ALL.iter().map(|family| family.model_type()).collect();
            bail!(
                "model architecture {model_type:?} is unsupported by this Ster build; use a Hugging Face checkpoint whose model_type is one of {}",
                families.join(", ")
            );
        };
        if raw.get("quantization_config").is_some() {
            bail!(
                "{} is a quantized checkpoint (it declares quantization_config); Ster maps unquantized safetensors only, so use the checkpoint it was quantized from",
                self.config.display()
            );
        }
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

/// Whether the repository lists a file, so an optional one is only fetched
/// when asking for it can succeed.
fn published(info: &hf_hub::api::RepoInfo, name: &str) -> bool {
    info.siblings.iter().any(|file| file.rfilename == name)
}

fn local_safetensors(root: &Path) -> Result<Vec<PathBuf>> {
    let mut weights = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("failed to list {}", root.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
            && !path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains("optimizer"))
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

fn eos_tokens(config: &LlamaConfig) -> BTreeSet<u32> {
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => [*token].into_iter().collect(),
        Some(LlamaEosToks::Multiple(tokens)) => tokens.iter().copied().collect(),
        None => BTreeSet::new(),
    }
}

/// The config with Python's non-finite float literals replaced by `null`.
///
/// Transformers writes configs with Python's `json`, which emits `Infinity`,
/// `-Infinity` and `NaN` for non-finite floats (Mamba-2's unbounded
/// `time_step_limit` is `[0.0, Infinity]`). JSON has no such tokens, so they
/// are rewritten — outside strings only — to `null`, which every reader of
/// such a key takes as "not stated". Bytes without them are returned as they
/// are.
fn finite_literals(bytes: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    const LITERALS: [&[u8]; 3] = [b"-Infinity", b"Infinity", b"NaN"];
    let mut output: Option<Vec<u8>> = None;
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            match (escaped, byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_string = false,
                _ => {}
            }
        } else if byte == b'"' {
            in_string = true;
        } else if let Some(literal) = LITERALS.iter().find(|literal| bytes[index..].starts_with(literal)) {
            let written = output.get_or_insert_with(|| bytes[..index].to_vec());
            written.extend_from_slice(b"null");
            index += literal.len();
            continue;
        }
        if let Some(written) = output.as_mut() {
            written.push(byte);
        }
        index += 1;
    }
    match output {
        Some(written) => std::borrow::Cow::Owned(written),
        None => std::borrow::Cow::Borrowed(bytes),
    }
}

/// The language model's config inside a multimodal one, and the prefix its
/// weights sit below.
///
/// Gemma 3's image-text checkpoints (`model_type` `gemma3`) and Mistral 3's
/// (`mistral3`, Ministral 3) nest the text decoder's config under
/// `text_config` and its weights under `language_model`; Gemma 4's
/// (`gemma4`, `gemma4_unified`) nest them the same way under
/// `model.language_model`; Step3's (`step3_vl`) nest the config the same
/// way and keep the text weights at the root. The vision and audio towers
/// beside it are never
/// read. The keys the nested config leaves to the outer one
/// (`eos_token_id`, `bos_token_id`, `tie_word_embeddings`,
/// `quantization_config`) are copied in. Any other config is returned as it
/// is, with no prefix.
fn language_model(outer: serde_json::Value) -> (serde_json::Value, &'static str) {
    const INHERITED: [&str; 4] = [
        "eos_token_id",
        "bos_token_id",
        "tie_word_embeddings",
        "quantization_config",
    ];
    let prefix = match outer.get("model_type").and_then(|value| value.as_str()) {
        Some("gemma3" | "mistral3") => "language_model",
        Some("gemma4" | "gemma4_unified") => "model.language_model",
        Some("step3_vl") => "",
        _ => return (outer, ""),
    };
    let Some(mut inner) = outer.get("text_config").cloned() else {
        return (outer, "");
    };
    if let Some(object) = inner.as_object_mut() {
        for key in INHERITED {
            if let (false, Some(value)) = (object.contains_key(key), outer.get(key)) {
                object.insert(key.to_owned(), value.clone());
            }
        }
    }
    (inner, prefix)
}
