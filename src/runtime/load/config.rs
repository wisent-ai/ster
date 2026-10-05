//! Reading a checkpoint's config file into the JSON the family reading
//! expects: Python's non-finite literals made JSON, a Mistral-format
//! `params.json` put in the Transformers shape, and a multimodal config's
//! language model taken out of its wrapper.

use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde_json::Value;

use super::mistral::{self, Layout};

/// The decoder's config as Transformers would state it, and the prefix its
/// weights sit below.
pub(super) fn read(path: &Path, layout: Layout) -> Result<(Value, &'static str)> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let outer: Value = serde_json::from_slice(&finite_literals(&bytes))
        .with_context(|| format!("invalid model config {}", path.display()))?;
    Ok(match layout {
        Layout::Transformers => language_model(outer),
        Layout::Mistral => (mistral::transformers_config(outer, path)?, ""),
    })
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
/// Gemma 3's image-text checkpoints (`model_type` `gemma3`), Mistral 3's
/// (`mistral3`, Ministral 3), Llama 4's (`llama4`) and Kimi-K3's (`kimi_k3`)
/// nest the text decoder's config under `text_config` and its weights under
/// `language_model`; Gemma 4's (`gemma4`, `gemma4_unified`), Qwen3.5's
/// (`qwen3_5`, `qwen3_5_moe`), MuseGlimmer's (`muse_glimmer`) and HyperCLOVA
/// X Vision V2's (`hyperclovax_vision_v2`) nest them the same way under
/// `model.language_model`, the head at the root or beside the decoder;
/// Step3's (`step3_vl`) nest the config the same way and keep the text
/// weights at the root. The vision and audio towers beside it are never
/// read. The keys the nested config leaves to the outer one
/// (`eos_token_id`, `bos_token_id`, `tie_word_embeddings`,
/// `quantization_config`) are copied in. Any other config is returned as it
/// is, with no prefix.
fn language_model(outer: Value) -> (Value, &'static str) {
    const INHERITED: [&str; 4] = [
        "eos_token_id",
        "bos_token_id",
        "tie_word_embeddings",
        "quantization_config",
    ];
    let prefix = match outer.get("model_type").and_then(|value| value.as_str()) {
        Some("gemma3" | "mistral3" | "llama4" | "kimi_k3") => "language_model",
        Some(
            "gemma3n" | "gemma4" | "gemma4_unified" | "qwen3_5" | "qwen3_5_moe" | "muse_glimmer" | "hyperclovax_vision_v2",
        ) => "model.language_model",
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
