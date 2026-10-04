//! Which decoder family a checkpoint is, read from its `config.json`.
//!
//! Every switch below is a key the checkpoint publishes; none is a value Ster
//! picks. A config that asks for something the decoder does not implement is
//! refused with the key that asked for it, instead of being loaded wrong.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Activation, Architecture};

/// The `model_type` values the decoder implements.
pub(super) const FAMILIES: &[&str] = &["llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2"];

/// What `model_type` adds to the Llama block, from the config's own keys.
pub(super) fn family(
    model_type: &str,
    raw: &Value,
    llama: &LlamaConfig,
    path: &Path,
) -> Result<Architecture> {
    let layers = llama.num_hidden_layers;
    let mut architecture = Architecture::llama(llama.hidden_size, llama.num_attention_heads);
    if let Some(head_dim) = whole(raw, "head_dim") {
        architecture.head_dim = head_dim;
        architecture.score_divisor = (head_dim as f64).sqrt();
    }
    let gemma = model_type.starts_with("gemma");
    if !gemma {
        if let Some(activation) = text(raw, "hidden_act").filter(|name| *name != "silu") {
            bail!(
                "{} declares hidden_act {activation:?}; Ster's {model_type} feed-forward gate is silu",
                path.display()
            );
        }
    }
    match model_type {
        "llama" | "mistral" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            // Mistral v0.2 and later publish `sliding_window: null`, which is
            // full attention on every layer.
            if model_type == "mistral" {
                architecture.sliding_window = whole(raw, "sliding_window");
                if architecture.sliding_window.is_some() {
                    architecture.sliding_layers = every_layer(layers, path)?;
                }
            }
        }
        "qwen2" | "qwen3" => {
            if model_type == "qwen2" {
                architecture.query_key_value_bias = true;
            } else {
                architecture.query_key_norm = true;
                architecture.query_key_value_bias = flag(raw, "attention_bias");
                architecture.output_bias = architecture.query_key_value_bias;
            }
            if flag(raw, "use_sliding_window") {
                architecture.sliding_window = whole(raw, "sliding_window");
                let from = whole(raw, "max_window_layers").unwrap_or(0);
                architecture.sliding_layers = every_layer(layers, path)? & !every_layer(from.min(layers), path)?;
            }
        }
        "gemma" | "gemma2" => {
            architecture.norm_offset = true;
            architecture.embedding_scale = true;
            architecture.activation = Activation::GeluTanh;
            if model_type == "gemma2" {
                architecture.sandwich_norms = true;
                architecture.sliding_window = whole(raw, "sliding_window");
                architecture.sliding_layers = even_layers(layers, path)?;
                if let Some(scalar) = number(raw, "query_pre_attn_scalar") {
                    architecture.score_divisor = scalar.sqrt();
                }
                architecture.attention_softcap = number(raw, "attn_logit_softcapping");
                architecture.final_softcap = number(raw, "final_logit_softcapping");
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    if let Some(types) = raw.get("layer_types").and_then(Value::as_array) {
        architecture.sliding_layers = listed_layers(types, layers, path)?;
    }
    if architecture.sliding_layers != 0 && architecture.sliding_window.is_none() {
        bail!(
            "{} makes some layers sliding-window but declares no sliding_window",
            path.display()
        );
    }
    Ok(architecture)
}

fn flag(raw: &Value, key: &str) -> bool {
    raw.get(key).and_then(Value::as_bool) == Some(true)
}

fn whole(raw: &Value, key: &str) -> Option<usize> {
    raw.get(key).and_then(Value::as_u64).map(|value| value as usize)
}

fn number(raw: &Value, key: &str) -> Option<f64> {
    raw.get(key).and_then(Value::as_f64)
}

fn text<'a>(raw: &'a Value, key: &str) -> Option<&'a str> {
    raw.get(key).and_then(Value::as_str)
}

/// A layer set needs one bit per layer.
fn fits(layers: usize, path: &Path) -> Result<()> {
    if layers > u128::BITS as usize {
        bail!(
            "{} has {layers} layers with sliding-window attention; Ster tracks the window per layer for at most {} layers",
            path.display(),
            u128::BITS
        );
    }
    Ok(())
}

fn every_layer(layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
    Ok((0..layers).fold(0, |set, layer| set | (1u128 << layer)))
}

fn even_layers(layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
    Ok((0..layers).step_by(2).fold(0, |set, layer| set | (1u128 << layer)))
}

/// `layer_types`, as newer configs list it: one entry per layer, either
/// `sliding_attention` or `full_attention`.
fn listed_layers(types: &[Value], layers: usize, path: &Path) -> Result<u128> {
    if types.len() != layers {
        bail!(
            "{} lists {} layer_types for {layers} layers",
            path.display(),
            types.len()
        );
    }
    fits(layers, path)?;
    let mut set = 0u128;
    for (layer, kind) in types.iter().enumerate() {
        match kind.as_str() {
            Some("sliding_attention") => set |= 1u128 << layer,
            Some("full_attention") => {}
            other => bail!(
                "{} declares layer {layer} as {other:?}; Ster implements sliding_attention and full_attention",
                path.display()
            ),
        }
    }
    Ok(set)
}
