//! ChatGLM's and Nemotron Puzzle's keys under Llama's names.

use serde_json::Value;

use crate::runtime::load::family::read::{flag, number};
use crate::runtime::load::family::rope::DEFAULT_ROPE_THETA;

/// The share of each ChatGLM head that rotates (vLLM's `chatglm.py`,
/// `"partial_rotary_factor": 0.5`).
pub(in crate::runtime::load::family) const CHATGLM_ROTARY_SHARE: f64 = 0.5;

/// ChatGLM's config, put in Llama's shape: its key-value groups
/// (`multi_query_group_num` under `multi_query_attention`), head width
/// (`kv_channels`), vocabulary (`padded_vocab_size`), the half of each head
/// that rotates, and the base `10000 · rope_ratio`.
pub(in crate::runtime::load::family) fn chatglm_keys(raw: &mut Value) {
    let groups = raw
        .get("multi_query_group_num")
        .filter(|_| flag(raw, "multi_query_attention"))
        .cloned();
    let heads = raw.get("num_attention_heads").cloned();
    let width = raw.get("kv_channels").cloned();
    let vocabulary = raw.get("padded_vocab_size").cloned();
    let base = DEFAULT_ROPE_THETA * number(raw, "rope_ratio").unwrap_or(1.0);
    let Some(object) = raw.as_object_mut() else {
        return;
    };
    if let Some(key_value_heads) = groups.or(heads) {
        object
            .entry("num_key_value_heads")
            .or_insert(key_value_heads);
    }
    if let Some(width) = width {
        object.entry("head_dim").or_insert(width);
    }
    if let Some(vocabulary) = vocabulary {
        object.entry("vocab_size").or_insert(vocabulary);
    }
    object
        .entry("partial_rotary_factor")
        .or_insert(Value::from(CHATGLM_ROTARY_SHARE));
    object.entry("rope_theta").or_insert(Value::from(base));
}

/// Nemotron Puzzle's config, put in Nemotron-H's shape: its
/// `layers_block_type` (`mamba`, `attention`, `mlp`, `moe`) becomes
/// `hybrid_override_pattern` (`M`, `*`, `-`, `E`), and the first `moe`
/// block's `num_experts_per_tok` and `moe_intermediate_size` stand in for
/// the top-level ones the per-layer `block_configs` override. A block type
/// it does not know is left as `?`, which the Nemotron-H reading refuses
/// by layer.
pub(in crate::runtime::load::family) fn nemotron_puzzle_keys(raw: &mut Value) {
    let pattern: Option<String> =
        raw.get("layers_block_type")
            .and_then(Value::as_array)
            .map(|kinds| {
                kinds
                    .iter()
                    .map(|kind| match kind.as_str() {
                        Some("mamba") => 'M',
                        Some("attention") => '*',
                        Some("mlp") => '-',
                        Some("moe") => 'E',
                        _ => '?',
                    })
                    .collect()
            });
    let first_moe = raw
        .get("block_configs")
        .and_then(Value::as_array)
        .and_then(|blocks| {
            blocks
                .iter()
                .find(|block| block.get("block_type").and_then(Value::as_str) == Some("moe"))
        })
        .cloned();
    let Some(object) = raw.as_object_mut() else {
        return;
    };
    if let Some(pattern) = pattern {
        object
            .entry("hybrid_override_pattern")
            .or_insert(Value::from(pattern));
    }
    if let Some(block) = first_moe {
        for key in ["num_experts_per_tok", "moe_intermediate_size"] {
            if let Some(value) = block.get(key).cloned() {
                object.entry(key).or_insert(value);
            }
        }
    }
}
