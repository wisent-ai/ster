//! Copying the keys other families spell otherwise under Llama's names before Candle parses them.

pub(in crate::runtime::load::family) mod aliases;
pub(in crate::runtime::load::family) mod glm;
pub(in crate::runtime::load::family) mod step;

use serde_json::Value;

use crate::runtime::load::family::classic::loops::deci_intermediate;
use crate::runtime::load::family::keys::glm::{chatglm_keys, nemotron_puzzle_keys};
use crate::runtime::load::family::keys::step::step3p5_keys;
use crate::runtime::load::family::read::whole;

/// OLMo 1's and DBRX's configs state no norm epsilon; their LayerNorms use
/// PyTorch's default, which this is.
pub(in crate::runtime::load::family) const OLMO_NORM_EPS: f64 = 1e-5;

/// GPT-J, GPT-2, GPT-BigCode and BLOOM make the feed-forward this many times
/// the residual width when the config states none, as their Transformers
/// blocks compute it.
pub(in crate::runtime::load::family) const GPT_INNER_PER_HIDDEN: u64 = 4;

/// Candle's Llama config reads Llama's key names. Other families spell the
/// same dimension otherwise (GPT-2's and GPT-J's `n_embd`, `n_layer`,
/// `n_head`, `n_positions`, `n_inner`; BLOOM's `n_embed`; OPT's `ffn_dim`;
/// GPT-NeoX's `rotary_emb_base`; the LayerNorm families' epsilon). Before the
/// config is parsed, the first spelling present is copied under Llama's name;
/// a key Llama's name already holds is left alone. OLMo 1, which states no
/// epsilon, gets the one its norm is defined with; the GPT families without
/// an inner width get four times the residual; GPT-BigCode's `multi_query`
/// becomes one key-value head.
pub(in crate::runtime::load) fn fill_llama_keys(raw: &mut Value, model_type: &str) {
    // DBRX keeps its attention and feed-forward settings in `attn_config` and
    // `ffn_config`; they are lifted beside the rest before any key is read,
    // never over a key the top level already states.
    if model_type == "dbrx" {
        let nested: Vec<(String, Value)> = ["attn_config", "ffn_config"]
            .iter()
            .filter_map(|section| raw.get(*section).and_then(Value::as_object))
            .flat_map(|section| {
                section
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
            })
            .collect();
        if let Some(object) = raw.as_object_mut() {
            for (key, value) in nested {
                object.entry(key).or_insert(value);
            }
        }
    }
    // Granite 4.0 without experts runs its `shared_mlp` as the feed-forward,
    // `shared_intermediate_size` wide; `intermediate_size` is then unused.
    if model_type == "granitemoehybrid" && whole(raw, "num_local_experts").unwrap_or(0) == 0 {
        let shared = raw
            .get("shared_intermediate_size")
            .filter(|width| width.is_u64())
            .cloned();
        if let (Some(width), Some(object)) = (shared, raw.as_object_mut()) {
            object.insert("intermediate_size".to_owned(), width);
        }
    }
    // Zamba2 ties its head to the embeddings whatever `tie_word_embeddings`
    // says (`Zamba2ForCausalLM._tied_weights_keys`), and its checkpoints
    // ship no `lm_head`.
    if model_type == "zamba2" {
        if let Some(object) = raw.as_object_mut() {
            object.insert("tie_word_embeddings".to_owned(), Value::Bool(true));
        }
    }
    // DeciLM states its feed-forward widths per layer and leaves
    // `intermediate_size` null; the widest stands in for the Llama key,
    // which no DeciLM layer reads.
    if model_type == "nemotron-nas" && raw.get("intermediate_size").is_none_or(Value::is_null) {
        let hidden = whole(raw, "hidden_size").unwrap_or(0);
        let widest = raw
            .get("block_configs")
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| block.get("ffn"))
                    .filter_map(|ffn| deci_intermediate(ffn, hidden))
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        if let Some(object) = raw.as_object_mut() {
            object.insert("intermediate_size".to_owned(), Value::from(widest));
        }
    }
    // MiMo-V2 states its full-attention layers' key-value heads as
    // `num_key_value_heads` and its sliding-window layers' as
    // `swa_num_key_value_heads`. Ster's base count is the sliding-window
    // one; the full-attention count moves to `num_global_key_value_heads`.
    if model_type.starts_with("mimo_v2") {
        let windowed = raw
            .get("swa_num_key_value_heads")
            .filter(|value| !value.is_null())
            .cloned();
        let full = raw.get("num_key_value_heads").cloned();
        if let (Some(windowed), Some(full), Some(object)) = (windowed, full, raw.as_object_mut()) {
            object.insert("num_global_key_value_heads".to_owned(), full);
            object.insert("num_key_value_heads".to_owned(), windowed);
        }
    }
    if model_type == "step3p5" {
        step3p5_keys(raw);
    }
    // LongCat-Flash's `num_layers` stored layers each hold two attention and
    // feed-forward pairs, which Ster runs as two layers.
    if model_type == "longcat_flash" || model_type == "longcat_flash_ngram" {
        let stored = whole(raw, "num_layers");
        if let (Some(stored), Some(object)) = (stored, raw.as_object_mut()) {
            object.insert("num_hidden_layers".to_owned(), Value::from(2 * stored));
        }
    }
    // Cohere2-MoE's `intermediate_size` is each expert's width; its dense
    // prefix layers are `prefix_dense_intermediate_size` wide, which becomes
    // the Llama key. Its norms are RMS norms when it states `rms_norm_eps`
    // and LayerNorms over `layer_norm_eps` otherwise (vLLM's
    // `select_norm_impl`), which `layer_norm` records before the epsilon
    // spellings merge.
    if model_type == "cohere2_moe" {
        let experts = raw.get("intermediate_size").cloned();
        let dense = raw
            .get("prefix_dense_intermediate_size")
            .filter(|width| !width.is_null())
            .cloned();
        let layer_norm = raw.get("rms_norm_eps").is_none_or(Value::is_null);
        if let Some(object) = raw.as_object_mut() {
            if let Some(experts) = experts {
                object.entry("moe_intermediate_size").or_insert(experts);
            }
            if let Some(dense) = dense {
                object.insert("intermediate_size".to_owned(), dense);
            }
            object.insert("layer_norm".to_owned(), Value::Bool(layer_norm));
        }
    }
    // Llama 4's `intermediate_size` is each expert's width and
    // `intermediate_size_mlp` its dense layers', which becomes the Llama
    // key.
    if model_type == "llama4_text" {
        let experts = raw.get("intermediate_size").cloned();
        let dense = raw
            .get("intermediate_size_mlp")
            .filter(|width| !width.is_null())
            .cloned();
        if let Some(object) = raw.as_object_mut() {
            if let Some(experts) = experts {
                object.entry("moe_intermediate_size").or_insert(experts);
            }
            if let Some(dense) = dense {
                object.insert("intermediate_size".to_owned(), dense);
            }
        }
    }
    if model_type == "minimax_m3_vl_text" {
        crate::runtime::load::family::custom::minimax_m3::fill_keys(raw);
    }
    // DeepSeek-V4 states only `moe_intermediate_size`, which Transformers
    // reads as `intermediate_size` too (`DeepseekV4Config.attribute_map`):
    // the shared expert's width.
    if model_type == "deepseek_v4" && raw.get("intermediate_size").is_none_or(Value::is_null) {
        let width = raw.get("moe_intermediate_size").cloned();
        if let (Some(width), Some(object)) = (width, raw.as_object_mut()) {
            object.insert("intermediate_size".to_owned(), width);
        }
    }
    if model_type == "nemotron_h_puzzle" {
        nemotron_puzzle_keys(raw);
    }
    if model_type == "chatglm" {
        chatglm_keys(raw);
    }
    // Gemma 3n states its feed-forward width per layer; a list of one width
    // is that width, and a list of several is left for the reading to
    // refuse by its key.
    if model_type == "gemma3n_text" {
        let uniform = raw
            .get("intermediate_size")
            .and_then(Value::as_array)
            .and_then(|widths| {
                widths
                    .first()
                    .filter(|first| widths.iter().all(|width| width == *first))
                    .cloned()
            });
        if let (Some(width), Some(object)) = (uniform, raw.as_object_mut()) {
            object.insert("intermediate_size".to_owned(), width);
        }
    }
    // Laguna states its full-attention layers' head count as
    // `num_attention_heads`; Ster's base count is its sliding-window
    // layers', read from `num_attention_heads_per_layer`, and the reading
    // gives the full-attention layers theirs.
    if model_type == "laguna" {
        let windowed = raw
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|kinds| {
                kinds
                    .iter()
                    .position(|kind| kind.as_str() == Some("sliding_attention"))
            })
            .and_then(|layer| {
                raw.get("num_attention_heads_per_layer")
                    .and_then(|counts| counts.get(layer))
                    .cloned()
            });
        if let (Some(heads), Some(object)) = (windowed, raw.as_object_mut()) {
            object.insert("num_attention_heads".to_owned(), heads);
        }
    }
    // openPangu-Ultra-MoE leaves its router's form to its config class:
    // sigmoid scores, renormalised (`PanguUltraMoEConfig`'s
    // `norm_topk_prob=True`, `MoEGate.forward`).
    if model_type == "pangu_ultra_moe" {
        if let Some(object) = raw.as_object_mut() {
            object
                .entry("scoring_func")
                .or_insert_with(|| Value::from("sigmoid"));
            object.entry("norm_topk_prob").or_insert(Value::Bool(true));
        }
    }
    aliases::copy_aliases(raw);
    aliases::derive_keys(raw, model_type);
}
