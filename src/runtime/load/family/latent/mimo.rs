//! MiMo-V2.

use std::path::Path;

use anyhow::Result;
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, GlobalAttention, Names, QkvLayout};
use crate::runtime::load::family::experts::deepseek::deepseek_experts;
use crate::runtime::load::family::read::layers::{every_layer, flagged_layers};
use crate::runtime::load::family::read::{first, flag, number, text, whole};

/// MiMo-V2 (`mimo_v2_flash`, `mimo_v2`): sliding-window layers where
/// `hybrid_layer_pattern` is one, with `swa_num_key_value_heads` key-value
/// heads (Ster's base count; the full-attention layers'
/// `num_key_value_heads` moves to `num_global_key_value_heads` as the config
/// is read), a learned sink per head under `add_swa_attention_sink_bias`
/// and their own base `swa_rope_theta`; full-attention layers carry sinks
/// under `add_full_attention_sink_bias`. Every head's query and key are
/// `head_dim` wide with the first `partial_rotary_factor` share rotated by
/// halves, its value `v_head_dim` wide and multiplied by
/// `attention_value_scale`. Query, key and value are separate, or stacked
/// in `qkv_proj` under `attention_projection_layout` `fused_qkv`. Layers
/// `moe_layer_freq` marks run DeepSeek-V3's routed experts.
pub(in crate::runtime::load::family) fn mimo_v2(
    raw: &Value,
    model_type: &str,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let pairs = [
        ("num_attention_heads", "swa_num_attention_heads"),
        ("head_dim", "swa_head_dim"),
        ("v_head_dim", "swa_v_head_dim"),
    ];
    for (full, windowed) in pairs {
        if let (Some(stated), Some(own)) = (whole(raw, full, path)?, whole(raw, windowed, path)?) {
            if stated != own {
                refused!(
                    Unimplemented,
                    in raw,
                    [windowed, full],
                    "{} gives its sliding-window layers {windowed} {own} and its full-attention layers {full} {stated}; Ster builds both kinds with one head count and width",
                    path.display()
                );
            }
        }
    }
    architecture.names = Names::MIMO_V2;
    architecture.qkv_layout = match text(raw, "attention_projection_layout", path)? {
        None => QkvLayout::Separate,
        Some("fused_qkv") => QkvLayout::Stacked,
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["attention_projection_layout"],
            "{} declares attention_projection_layout {other:?}; Ster implements separate projections and fused_qkv",
            path.display()
        ),
    };
    architecture.query_key_value_bias = flag(raw, "attention_bias", path)?;
    let windowed = flagged_layers(raw, "hybrid_layer_pattern", layers, path)?;
    architecture.sliding_layers = windowed;
    architecture.sliding_window = first(raw, &["sliding_window_size", "sliding_window"], path)?;
    let full = every_layer(layers, path)? & !windowed;
    architecture.attention_sinks = if flag(raw, "add_swa_attention_sink_bias", path)? {
        windowed
    } else {
        0
    } | if flag(raw, "add_full_attention_sink_bias", path)? {
        full
    } else {
        0
    };
    architecture.local_rope_theta = number(raw, "swa_rope_theta", path)?
        .filter(|base| *base != f64::from(llama.rope_theta))
        .map(|base| base as f32);
    architecture.value_head_dim = whole(raw, "v_head_dim", path)?;
    architecture.value_scale = number(raw, "attention_value_scale", path)?;
    architecture.global_attention = Some(GlobalAttention {
        heads: None,
        head_dim: architecture.head_dim,
        key_value_heads: whole(raw, "num_global_key_value_heads", path)?,
        rotary_dim: architecture.rotary_dim,
        key_is_value: false,
    });
    architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
    Ok(())
}
