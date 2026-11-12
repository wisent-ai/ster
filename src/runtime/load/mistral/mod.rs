//! Mistral's own checkpoint format: `params.json` beside
//! `consolidated*.safetensors`, as Mistral Large 3 alone is published. The
//! config is put in the Transformers shape the rest of loading reads, and
//! the decoder's tensor names are answered from Mistral's, following vLLM's
//! `transformers_utils/configs/mistral.py` and the weight mappers of its
//! `llama.py` and `mistral_large_3.py`.

use std::path::Path;

use anyhow::Result;
use serde_json::{Map, Value};

mod blocks;
mod eos;
mod names;

use crate::model::keys::within;
use blocks::{experts, rope_yarn, sliding_window};

pub use eos::end_of_sequence;

/// How a checkpoint names its config and weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `config.json` and the Transformers tensor names.
    Transformers,
    /// `params.json`, `consolidated*.safetensors` and Mistral's tensor names.
    Mistral,
    /// `config.json` and the base model's tokenizer beside one `*.gguf` file
    /// holding llama.cpp's tensor names (`super::gguf`).
    Gguf,
}

// What vLLM's `_remap_general_mistral_args` and `_remap_mistral_yarn_args`
// give a Mistral-format config that leaves a key out:
// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/mistral.py
/// `max_position_embeddings`, `128_000`.
const MISTRAL_DEFAULT_CONTEXT: u64 = 128_000;
/// `hidden_act` without `activation`.
const MISTRAL_ACTIVATION: &str = "silu";
/// `tie_word_embeddings` without `tied_embeddings`.
const MISTRAL_TIED: bool = false;

impl Layout {
    /// The config file this layout reads.
    pub fn config_file(self) -> &'static str {
        match self {
            Self::Transformers | Self::Gguf => "config.json",
            Self::Mistral => "params.json",
        }
    }

    /// The layout as a refusal names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Transformers => "Transformers (config.json)",
            Self::Mistral => "Mistral (params.json)",
            Self::Gguf => "GGUF (config.json beside one .gguf file)",
        }
    }

    /// The weight files this layout reads, as a refusal names them.
    pub fn weights_kind(self) -> &'static str {
        match self {
            Self::Transformers => "*.safetensors",
            Self::Mistral => "consolidated*.safetensors",
            Self::Gguf => ".gguf",
        }
    }

    /// The weight file a merge writes, named so the merged directory
    /// resolves under the same layout. A GGUF base is refused by the merge
    /// before it gets here: the merge writes the base's tensors back as
    /// safetensors, and a GGUF holds quantized ones.
    pub fn merged_weights(self) -> &'static str {
        match self {
            Self::Transformers | Self::Gguf => "model.safetensors",
            Self::Mistral => "consolidated.safetensors",
        }
    }

    /// Whether a published weight file holds this layout's weights. A
    /// repository may publish both safetensors layouts (Mistral 7B v0.3);
    /// each reads its own files only, so the weights are never mapped twice.
    pub fn owns(self, file_name: &str) -> bool {
        if self == Self::Gguf {
            return file_name.ends_with(".gguf");
        }
        let consolidated = file_name.starts_with("consolidated");
        file_name.ends_with(".safetensors")
            && !file_name.contains("optimizer")
            && !file_name.contains("training_args")
            && consolidated == (self == Self::Mistral)
    }

    /// The name a tensor the decoder asks for by its Transformers name is
    /// stored under.
    pub fn stored_name(self, name: &str) -> String {
        match self {
            Self::Transformers => name.to_owned(),
            Self::Mistral => names::stored_name(name),
            Self::Gguf => super::gguf::stored_name(name),
        }
    }
}

/// `params` put in the Transformers shape. Mistral Large 3 (latent
/// attention and experts beside shared ones) becomes `deepseek_v3` with
/// softmax scores, renormalised; latent attention without experts becomes a
/// dense `deepseek_v3`; a dense model becomes `mistral`, rotating adjacent
/// pairs as Mistral's own weights are laid out. Experts without shared ones
/// (Mixtral) are refused: their Mistral-format names are not read.
pub fn transformers_config(params: Value, path: &Path) -> Result<Value> {
    let mut config = match params {
        Value::Object(config) => config,
        other => return Err(crate::failures::model::config_not_object(path, &other)),
    };
    rename(&mut config, "dim", "hidden_size");
    rename(&mut config, "norm_eps", "rms_norm_eps");
    rename(&mut config, "n_kv_heads", "num_key_value_heads");
    rename(&mut config, "n_layers", "num_hidden_layers");
    rename(&mut config, "n_heads", "num_attention_heads");
    rename(&mut config, "hidden_dim", "intermediate_size");
    let activation = match config.remove("activation") {
        Some(stated) => stated,
        None => Value::from(MISTRAL_ACTIVATION),
    };
    config.insert("hidden_act".to_owned(), activation);
    let tied = match config.remove("tied_embeddings") {
        Some(stated) => stated,
        None => Value::Bool(MISTRAL_TIED),
    };
    config.insert("tie_word_embeddings".to_owned(), tied);
    config
        .entry("max_position_embeddings")
        .or_insert(Value::from(MISTRAL_DEFAULT_CONTEXT));
    if let Some(quantization) = config.remove("quantization") {
        config.insert("quantization_config".to_owned(), quantization);
    }
    sliding_window(&mut config, path)?;
    let latent = config
        .get("qk_nope_head_dim")
        .is_some_and(|width| !width.is_null());
    let model_type = match config.remove("moe") {
        Some(Value::Object(moe)) => {
            let section = Value::Object(moe.clone());
            let shared = within::<usize>(&section, "moe", "num_shared_experts", path)?;
            if !shared.is_some_and(|shared| shared > 0) {
                refused!(
                    Unimplemented,
                    ["moe" => Value::Object(moe.clone())],
                    "{} declares experts without shared ones (Mixtral's layout); Ster reads Mistral's own format for dense models and Mistral Large 3 only, so use the Transformers conversion",
                    path.display()
                );
            }
            experts(&mut config, moe);
            "deepseek_v3"
        }
        Some(Value::Null) | None if latent => "deepseek_v3",
        Some(Value::Null) | None => {
            config.insert("rope_interleave".to_owned(), Value::Bool(true));
            "mistral"
        }
        Some(other) => refused!(
            Malformed,
            ["moe" => other],
            "{} declares moe {other}, which is not an object",
            path.display()
        ),
    };
    config.insert("model_type".to_owned(), Value::from(model_type));
    if let Some(yarn) = config.remove("yarn").filter(|yarn| !yarn.is_null()) {
        rope_yarn(&mut config, &yarn, path)?;
    }
    Ok(Value::Object(config))
}

fn rename(config: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(value) = config.remove(from) {
        config.insert(to.to_owned(), value);
    }
}
