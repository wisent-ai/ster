//! Candle's Llama config, parsed from a config whose keys
//! [`fill_llama_keys`](super::super::fill_llama_keys) has put under Llama's
//! names, with every key it cannot do without refused by name first.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::runtime::load::family::keys::aliases::ALIASES;

/// The keys Candle's `LlamaConfig` declares without a default, and the type
/// each must have.
const REQUIRED: &[(&str, &str)] = &[
    ("hidden_size", "a whole number"),
    ("intermediate_size", "a whole number"),
    ("vocab_size", "a whole number"),
    ("num_hidden_layers", "a whole number"),
    ("num_attention_heads", "a whole number"),
    ("rms_norm_eps", "a number"),
    ("max_position_embeddings", "a whole number"),
];

/// The `model_type` decoder's Llama config from `raw`, or the refusal of
/// the first key Candle needs that `raw` leaves out or mistypes, naming the
/// key, the spellings read in its place, and the config at `path`.
pub(in crate::runtime::load) fn llama_config(
    raw: &Value,
    model_type: &str,
    path: &Path,
) -> Result<LlamaConfig> {
    for (key, kind) in REQUIRED {
        let fits = match raw.get(*key) {
            None | Some(Value::Null) => {
                bail!(
                    "{} states no {key}{}; Ster needs the {model_type} decoder's {key} ({kind}), so add {key} to the config",
                    path.display(),
                    spellings(key)
                )
            }
            Some(value) if *kind == "a number" => value.is_number(),
            Some(value) => value.is_u64(),
        };
        if !fits {
            bail!(
                "{} states {key} as {}, which is not {kind}; state the {model_type} decoder's {key} as {kind}",
                path.display(),
                raw[*key]
            );
        }
    }
    serde_json::from_value(raw.clone())
        .with_context(|| format!("invalid {model_type} config {}", path.display()))
}

/// The other spellings [`fill_llama_keys`](super::super::fill_llama_keys)
/// reads `key` from, as a refusal lists them, or nothing when it reads none.
fn spellings(key: &str) -> String {
    match ALIASES.iter().find(|(llama, _)| *llama == key) {
        Some((_, others)) => format!(" (nor {})", others.join(", ")),
        None => String::new(),
    }
}
