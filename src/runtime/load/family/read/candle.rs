//! Candle's Llama config, parsed from a config whose keys
//! [`fill_llama_keys`](super::super::fill_llama_keys) has put under Llama's
//! names, with every key it cannot do without refused by name first.

use std::path::Path;

use anyhow::Result;
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use super::stated;
use crate::failures::{model::config as causes, refuse};
use crate::runtime::load::family::keys::aliases::ALIASES;

/// The keys Candle's `LlamaConfig` declares without a default, and whether
/// each is a whole number (the others are numbers).
const REQUIRED: &[(&str, bool)] = &[
    ("hidden_size", true),
    ("intermediate_size", true),
    ("vocab_size", true),
    ("num_hidden_layers", true),
    ("num_attention_heads", true),
    ("rms_norm_eps", false),
    ("max_position_embeddings", true),
];

/// The `model_type` decoder's Llama config from `raw`, or the refusal of
/// the first key Candle needs that `raw` leaves out or mistypes, naming the
/// key, the spellings read in its place, and the config at `path`.
pub(in crate::runtime::load) fn llama_config(
    raw: &Value,
    model_type: &str,
    path: &Path,
) -> Result<LlamaConfig> {
    let shown = path.display().to_string();
    for (key, whole) in REQUIRED {
        let (present, kind) = if *whole {
            (stated::<usize>(raw, key, path)?.is_some(), "a whole number")
        } else {
            (stated::<f64>(raw, key, path)?.is_some(), "a number")
        };
        if !present {
            return Err(refuse(
                &causes::KEY_ABSENT,
                &[
                    ("path", &shown),
                    ("key", key),
                    ("spellings", &spellings(key)),
                    ("model_type", model_type),
                    ("kind", kind),
                ],
            ));
        }
    }
    serde_json::from_value(raw.clone()).map_err(|error| {
        refuse(
            &causes::UNDECODABLE,
            &[
                ("path", &shown),
                ("model_type", model_type),
                ("error", &error.to_string()),
            ],
        )
    })
}

/// The other spellings [`fill_llama_keys`](super::super::fill_llama_keys)
/// reads `key` from, as a refusal lists them, or nothing when it reads none.
fn spellings(key: &str) -> String {
    match ALIASES.iter().find(|(llama, _)| *llama == key) {
        Some((_, others)) => format!(" (nor {})", others.join(", ")),
        None => String::new(),
    }
}
