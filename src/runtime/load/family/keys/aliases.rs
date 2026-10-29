//! The other spellings of Llama's keys, and the widths and head counts
//! families leave Candle's parse to derive.

use serde_json::Value;

use crate::runtime::load::family::keys::{GPT_INNER_PER_HIDDEN, OLMO_NORM_EPS};
use crate::runtime::load::family::read::flag;

/// Each Llama key and the spellings other families state it under, the
/// first present of which is copied under Llama's name.
pub(in crate::runtime::load::family) const ALIASES: &[(&str, &[&str])] = &[
    (
        "rms_norm_eps",
        &[
            "layer_norm_eps",
            "norm_epsilon",
            "norm_eps",
            "layer_norm_epsilon",
            "layernorm_epsilon",
        ],
    ),
    ("hidden_size", &["n_embd", "n_embed", "d_model"]),
    ("num_hidden_layers", &["n_layer", "n_layers", "num_layers"]),
    ("num_attention_heads", &["n_head", "n_heads"]),
    (
        "num_key_value_heads",
        &["kv_n_heads", "num_attention_groups"],
    ),
    ("head_dim", &["attention_head_dim"]),
    (
        "max_position_embeddings",
        &[
            "n_positions",
            "max_seq_len",
            "seq_length",
            "model_max_length",
        ],
    ),
    (
        "intermediate_size",
        &["n_inner", "ffn_dim", "ffn_hidden_size"],
    ),
    ("rope_theta", &["rotary_emb_base"]),
    (
        "num_experts_per_tok",
        &[
            "moe_k",
            "moe_top_k",
            "moe_topk",
            "num_experts_per_token",
            "top_k_experts",
        ],
    ),
    // openPangu-Ultra-MoE's names for DeepSeek's latent attention and
    // experts.
    ("kv_lora_rank", &["attention_kv_lora_dim"]),
    ("q_lora_rank", &["attention_q_lora_dim"]),
    ("qk_nope_head_dim", &["attention_qk_dim"]),
    ("qk_rope_head_dim", &["attention_qk_rope_dim"]),
    ("v_head_dim", &["attention_v_dim"]),
    ("n_routed_experts", &["num_routed_experts"]),
    ("n_shared_experts", &["num_shared_experts"]),
    ("first_k_dense_replace", &["num_dense_layers"]),
];

/// Copies the first stated spelling of each [`ALIASES`] key under Llama's
/// name, never over a key Llama's name already holds.
pub(in crate::runtime::load::family) fn copy_aliases(raw: &mut Value) {
    for (llama, spellings) in ALIASES {
        if raw.get(*llama).is_some_and(|value| !value.is_null()) {
            continue;
        }
        let found = spellings
            .iter()
            .find_map(|key| raw.get(*key).filter(|value| !value.is_null()).cloned());
        if let (Some(value), Some(object)) = (found, raw.as_object_mut()) {
            object.insert((*llama).to_owned(), value);
        }
    }
}

/// The Llama keys a family's config leaves to be derived from others.
pub(in crate::runtime::load::family) fn derive_keys(raw: &mut Value, model_type: &str) {
    let missing = |raw: &Value, key: &str| raw.get(key).is_none_or(Value::is_null);
    let mut defaults: Vec<(&str, Value)> = Vec::new();
    // OLMo 1 and DBRX state no epsilon; their LayerNorms use PyTorch's.
    if matches!(model_type, "olmo" | "dbrx") && missing(raw, "rms_norm_eps") {
        defaults.push(("rms_norm_eps", Value::from(OLMO_NORM_EPS)));
    }
    let four_times = matches!(
        model_type,
        "gptj" | "gpt2" | "gpt_bigcode" | "bloom" | "falcon"
    );
    if four_times && missing(raw, "intermediate_size") {
        if let Some(hidden) = raw.get("hidden_size").and_then(Value::as_u64) {
            defaults.push((
                "intermediate_size",
                Value::from(GPT_INNER_PER_HIDDEN * hidden),
            ));
        }
    }
    // MPT states its feed-forward as a multiple of the model width.
    if model_type == "mpt" && missing(raw, "intermediate_size") {
        let width = raw.get("hidden_size").and_then(Value::as_u64);
        let ratio = raw.get("expansion_ratio").and_then(Value::as_u64);
        if let (Some(width), Some(ratio)) = (width, ratio) {
            defaults.push(("intermediate_size", Value::from(width * ratio)));
        }
    }
    // LFM2 states its feed-forward as `block_ff_dim`, which Transformers
    // shrinks to two thirds, scales by `block_ffn_dim_multiplier` and rounds
    // up to `block_multiple_of` when `block_auto_adjust_ff_dim` is set.
    if model_type == "lfm2" && missing(raw, "intermediate_size") {
        if let Some(width) = raw.get("block_ff_dim").and_then(Value::as_f64) {
            let width = if flag(raw, "block_auto_adjust_ff_dim") {
                const SWIGLU_SHARE: f64 = 2.0 / 3.0;
                let shrunk = (width * SWIGLU_SHARE).trunc();
                let scaled = match raw.get("block_ffn_dim_multiplier").and_then(Value::as_f64) {
                    Some(multiplier) => (shrunk * multiplier).trunc(),
                    None => shrunk,
                };
                let multiple = raw
                    .get("block_multiple_of")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                (scaled / multiple).ceil() * multiple
            } else {
                width
            };
            defaults.push(("intermediate_size", Value::from(width as u64)));
        }
    }
    // Mamba states its inner width as `expand` times the model width when it
    // leaves `intermediate_size` out.
    let state_space_only = matches!(model_type, "mamba" | "falcon_mamba" | "mamba2");
    if state_space_only && missing(raw, "intermediate_size") {
        let width = raw.get("hidden_size").and_then(Value::as_u64);
        let expand = raw.get("expand").and_then(Value::as_u64);
        if let (Some(width), Some(expand)) = (width, expand) {
            defaults.push(("intermediate_size", Value::from(width * expand)));
        }
    }
    // A state-space model has no attention heads and no position table;
    // Candle's config needs both counts, and these stand in, read by nothing.
    if state_space_only {
        if missing(raw, "num_attention_heads") {
            defaults.push(("num_attention_heads", Value::from(1u64)));
        }
        if missing(raw, "max_position_embeddings") {
            defaults.push(("max_position_embeddings", Value::from(1u64)));
        }
    }
    if model_type == "gpt_bigcode" && flag(raw, "multi_query") {
        defaults.push(("num_key_value_heads", Value::from(1u64)));
    }
    // Falcon's new decoder architecture groups query heads under
    // `num_kv_heads`; the older multi-query models have one key-value head.
    if model_type == "falcon" {
        if flag(raw, "new_decoder_architecture") {
            if let Some(groups) = raw.get("num_kv_heads").cloned() {
                defaults.push(("num_key_value_heads", groups));
            }
        } else if flag(raw, "multi_query") {
            defaults.push(("num_key_value_heads", Value::from(1u64)));
        }
    }
    if let Some(object) = raw.as_object_mut() {
        for (key, value) in defaults {
            object.insert(key.to_owned(), value);
        }
    }
}
