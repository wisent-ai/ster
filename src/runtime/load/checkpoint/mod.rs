//! Which files a checkpoint is, resolved once from a local directory or the
//! hub, and the config questions every loader asks of them.

mod resolve;

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Result, bail};
use candle_transformers::models::llama::{Config, LlamaConfig, LlamaEosToks};

use crate::{chat, model::Architecture};

use super::{
    config,
    family::{Family, family, fill_llama_keys, llama_config, take_rope_scaling},
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
            if let Some(eos) =
                mistral::end_of_sequence(self.tokenizer_config.as_deref(), &self.tokenizer)?
            {
                object
                    .entry("eos_token_id")
                    .or_insert(serde_json::Value::from(eos));
            }
        }
        let (model_type, found) = supported(&raw, &self.config)?;
        let model_type = model_type.as_str();
        let scaling = take_rope_scaling(&mut raw, &self.config)?;
        fill_llama_keys(&mut raw, model_type);
        let mut llama: LlamaConfig = llama_config(&raw, model_type, &self.config)?;
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
    let families: Vec<&str> = Family::ALL
        .iter()
        .map(|family| family.model_type())
        .collect();
    let families = families.join(", ");
    let model_type = match raw.get("model_type") {
        Some(serde_json::Value::String(stated)) => stated.clone(),
        None | Some(serde_json::Value::Null) => match Family::of_architectures(raw) {
            Some(found) => found.model_type().to_owned(),
            None => bail!(
                "{} names no model type: it states no model_type, and its architectures ({}) name no model class Ster reads in its place; use a Hugging Face checkpoint whose model_type is one of {families}",
                path.display(),
                match raw.get("architectures") {
                    Some(architectures) => architectures.to_string(),
                    None => "absent".to_owned(),
                }
            ),
        },
        Some(other) => bail!(
            "{} states model_type as {other}, which is not a string; use a Hugging Face checkpoint whose model_type is one of {families}",
            path.display()
        ),
    };
    let Some(found) = Family::of(&model_type) else {
        bail!(
            "model architecture {model_type:?} is unsupported by this Ster build; use a Hugging Face checkpoint whose model_type is one of {families}"
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

fn eos_tokens(config: &LlamaConfig) -> BTreeSet<u32> {
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => [*token].into_iter().collect(),
        Some(LlamaEosToks::Multiple(tokens)) => tokens.iter().copied().collect(),
        None => BTreeSet::new(),
    }
}
