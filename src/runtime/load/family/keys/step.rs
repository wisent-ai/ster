//! Step-3.5's keys under Llama's names.

use serde_json::Value;

use crate::runtime::load::family::read::whole;

/// Step 3.5's config, put in the shape the rest of the reading expects. Its
/// per-layer lists also cover the multi-token-prediction layers past
/// `num_hidden_layers`, which Ster does not run, so they are cut to the
/// decoder's layers; a list `rope_theta` becomes `layer_rope_theta`, with
/// the first full-attention layer's base as `rope_theta`; and the
/// sliding-window layers' head counts in `attention_other_setting` become
/// the base counts, the full-attention ones moving to
/// `num_global_attention_heads` and `num_global_key_value_heads`.
pub(in crate::runtime::load::family) fn step3p5_keys(raw: &mut Value) {
    let layers = whole(raw, "num_hidden_layers").unwrap_or(0);
    let Some(object) = raw.as_object_mut() else {
        return;
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
        let first_full = object
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|kinds| {
                kinds
                    .iter()
                    .position(|kind| kind.as_str() == Some("full_attention"))
            })
            .unwrap_or(0);
        if let Some(base) = bases.get(first_full).cloned() {
            object.insert("rope_theta".to_owned(), base);
        }
        object.insert("layer_rope_theta".to_owned(), Value::Array(bases));
    }
    let Some(other) = object
        .get("attention_other_setting")
        .filter(|setting| {
            setting.get("attention_type").and_then(Value::as_str) == Some("sliding_attention")
        })
        .cloned()
    else {
        return;
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
}
