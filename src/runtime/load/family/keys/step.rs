//! Step-3.5's keys under Llama's names.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::runtime::load::family::read::{list, stated, text, whole_or};

/// Step 3.5's config, put in the shape the rest of the reading expects. Its
/// per-layer lists also cover the multi-token-prediction layers past
/// `num_hidden_layers`, which Ster does not run, so they are cut to the
/// decoder's layers; a list `rope_theta` becomes `layer_rope_theta`, with
/// the first full-attention layer's base as `rope_theta`; and the
/// sliding-window layers' head counts in `attention_other_setting` become
/// the base counts, the full-attention ones moving to
/// `num_global_attention_heads` and `num_global_key_value_heads`.
pub(in crate::runtime::load::family) fn step3p5_keys(raw: &mut Value, path: &Path) -> Result<()> {
    // Candle's parse refuses an absent layer count by name after this.
    let layers = whole_or(raw, "num_hidden_layers", 0, path)?;
    let mut first_full = 0;
    for (layer, kind) in list(raw, "layer_types", path)?.into_iter().flatten().enumerate() {
        if kind.as_str() == Some("full_attention") {
            first_full = layer;
            break;
        }
    }
    let sliding_other = match stated::<&serde_json::Map<String, Value>>(
        raw,
        "attention_other_setting",
        path,
    )? {
        Some(_) => {
            text(raw, "attention_other_setting.attention_type", path)? == Some("sliding_attention")
        }
        None => false,
    };
    let Some(object) = raw.as_object_mut() else {
        return Ok(());
    };
    let per_layer = [
        "layer_types",
        "rope_theta",
        "partial_rotary_factors",
        "swiglu_limits",
        "swiglu_limits_shared",
        "use_rope_layers",
    ];
    for key in per_layer {
        if let Some(Value::Array(list)) = object.get_mut(key) {
            list.truncate(layers);
        }
    }
    if let Some(Value::Array(bases)) = object.get("rope_theta").cloned() {
        if let Some(base) = bases.get(first_full).cloned() {
            object.insert("rope_theta".to_owned(), base);
        }
        object.insert("layer_rope_theta".to_owned(), Value::Array(bases));
    }
    if !sliding_other {
        return Ok(());
    }
    let Some(other) = object.get("attention_other_setting").cloned() else {
        return Ok(());
    };
    let full_heads = object.get("num_attention_heads").cloned();
    let full_groups = object
        .get("num_attention_groups")
        .or_else(|| object.get("num_key_value_heads"))
        .cloned();
    let windowed_heads = other.get("num_attention_heads").cloned();
    let windowed_groups = other.get("num_attention_groups").cloned();
    if let (Some(heads), Some(groups), Some(own_heads), Some(own_groups)) =
        (full_heads, full_groups, windowed_heads, windowed_groups)
    {
        object.insert("num_global_attention_heads".to_owned(), heads);
        object.insert("num_global_key_value_heads".to_owned(), groups);
        object.insert("num_attention_heads".to_owned(), own_heads);
        object.insert("num_key_value_heads".to_owned(), own_groups);
    }
    Ok(())
}
