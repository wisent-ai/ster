//! DeciLM's per-layer plans and the looped decoders: Nanbeige and IQuest-LoopCoder.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, LayerPlan, Loops, Names, QueryKeyNorm};
use crate::runtime::load::family::At;
use crate::runtime::load::family::read::{flag, number, text, whole};

/// What the families this file reads add to the Llama block; [`super::super::family`]
/// sends only their `model_type` values here.
pub(in crate::runtime::load::family) fn configure(
    at: &At,
    architecture: &mut Architecture,
) -> Result<()> {
    let At {
        model_type,
        raw,
        llama,
        layers,
        path,
        ..
    } = *at;
    match model_type {
        "nemotron-nas" => {
            // DeciLM (Llama-3.3-Nemotron Super and Ultra): Llama's block,
            // each layer's own `block_configs` entry giving its key-value
            // heads (`num_attention_heads / n_heads_in_group`) and
            // feed-forward width (`ffn_mult`), either half possibly a
            // no-op. Linear replacements, attention windows and sinks, and
            // sparsified layers are refused.
            architecture.query_key_value_bias = flag(raw, "attention_bias") || flag(raw, "bias");
            architecture.output_bias = architecture.query_key_value_bias;
            if let Some(bias) = raw.get("qkv_bias").and_then(Value::as_bool) {
                architecture.query_key_value_bias = bias;
            }
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.layer_plans = Some(deci_plans(raw, layers, llama, path)?);
        }
        "nanbeige" => {
            // Nanbeige: Llama's block, per-head `q_layernorm` and
            // `k_layernorm` under `qk_layernorm`, and the stored layers run
            // `num_loops` times, the final norm closing every pass unless
            // `skip_loop_final_norm`. Its n-gram embeddings, hyper-connections,
            // split loops, shared loop caches and depth attention are
            // refused.
            for key in [
                "enable_hyper_connection",
                "enable_mhc",
                "enable_double_loop_split",
                "loop_share_kv",
                "enable_depth_attention",
            ] {
                if flag(raw, key) {
                    bail!(
                        "{} turns on Nanbeige's {key}, which Ster does not implement",
                        path.display()
                    );
                }
            }
            if raw
                .get("ngram_vocab_size_ratio")
                .is_some_and(|value| !value.is_null())
            {
                bail!(
                    "{} declares Nanbeige's n-gram embeddings (ngram_vocab_size_ratio), which Ster does not implement",
                    path.display()
                );
            }
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            if flag(raw, "qk_layernorm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.names = Names::NANBEIGE;
            }
            let count = whole(raw, "num_loops").unwrap_or(1).max(1);
            if count > 1 {
                architecture.loops = Some(Loops {
                    physical: layers,
                    count,
                    norm_between: !flag(raw, "skip_loop_final_norm"),
                    gate_window: None,
                });
            }
        }
        "iquestloopcoder" => {
            // IQuest-LoopCoder: Llama's block with `mlp_bias`, its stored
            // layers run `loop_num` times; each later pass mixes global
            // attention over the first pass's keys and values with local
            // attention over its own within `loop_window_size`, gated per
            // head by `model.gate_projections.{layer}` of the rotated query.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            let count = whole(raw, "loop_num").unwrap_or(LOOP_CODER_LOOPS).max(1);
            let window = whole(raw, "loop_window_size").unwrap_or(LOOP_CODER_WINDOW);
            if count > 1 {
                architecture.loops = Some(Loops {
                    physical: layers,
                    count,
                    norm_between: false,
                    gate_window: Some(window),
                });
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// IQuest-LoopCoder's defaults when its config leaves them out: two passes
/// and a 64-position local window (vLLM's `iquest_loopcoder.py`,
/// `getattr(config, "loop_num", 2)` and `getattr(config,
/// "loop_window_size", 64)`).
pub(in crate::runtime::load::family) const LOOP_CODER_LOOPS: usize = 2;

pub(in crate::runtime::load::family) const LOOP_CODER_WINDOW: usize = 64;

/// DeciLM rounds every feed-forward width up to a multiple of this
/// (`_find_multiple(intermediate_size, 256)` in vLLM's `nemotron_nas.py`
/// and the checkpoints' `modeling_decilm.py`).
pub(in crate::runtime::load::family) const DECI_WIDTH_MULTIPLE: usize = 256;

/// A DeciLM `ffn` entry's width: its `intermediate_size`, or
/// `2 · ffn_mult · hidden / 3` truncated and rounded up to a multiple of
/// [`DECI_WIDTH_MULTIPLE`].
pub(in crate::runtime::load::family) fn deci_intermediate(
    ffn: &Value,
    hidden: usize,
) -> Option<usize> {
    if let Some(width) = whole(ffn, "intermediate_size") {
        return Some(width);
    }
    let multiplier = number(ffn, "ffn_mult")?;
    let width = (2.0 * multiplier * hidden as f64 / 3.0) as usize;
    Some(width.div_ceil(DECI_WIDTH_MULTIPLE) * DECI_WIDTH_MULTIPLE)
}

/// DeciLM's `block_configs`, one plan per layer.
pub(in crate::runtime::load::family) fn deci_plans(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    path: &Path,
) -> Result<Vec<LayerPlan>> {
    let Some(blocks) = raw.get("block_configs").and_then(Value::as_array) else {
        bail!(
            "{} declares a DeciLM model without block_configs",
            path.display()
        );
    };
    if blocks.len() != layers {
        bail!(
            "{} lists {} block_configs for {layers} layers",
            path.display(),
            blocks.len()
        );
    }
    let stated = |section: &Value, key: &str| {
        section
            .get(key)
            .is_some_and(|value| !value.is_null() && value != false)
    };
    blocks
        .iter()
        .enumerate()
        .map(|(layer, block)| {
            let (Some(attention), Some(ffn)) = (block.get("attention"), block.get("ffn")) else {
                bail!("{} gives layer {layer} no attention or ffn block config", path.display());
            };
            for (section, name) in [(attention, "attention"), (ffn, "ffn")] {
                for key in ["replace_with_linear", "sparsify"] {
                    if stated(section, key) {
                        bail!("{} sets layer {layer}'s {name} {key}, which Ster does not implement", path.display());
                    }
                }
            }
            for key in ["window_length", "num_sink_tokens"] {
                if stated(attention, key) {
                    bail!("{} sets layer {layer}'s attention {key}, which Ster does not implement", path.display());
                }
            }
            if text(ffn, "hidden_act").is_some_and(|act| Some(act) != text(raw, "hidden_act")) {
                bail!(
                    "{} gives layer {layer} its own feed-forward hidden_act; Ster runs one activation in every layer",
                    path.display()
                );
            }
            let key_value_heads = if flag(attention, "no_op") {
                None
            } else {
                let Some(group) = whole(attention, "n_heads_in_group").filter(|group| *group > 0) else {
                    bail!("{} gives layer {layer}'s attention no n_heads_in_group", path.display());
                };
                if llama.num_attention_heads % group != 0 {
                    bail!(
                        "{} groups layer {layer}'s {} heads by {group}, which does not divide them",
                        path.display(),
                        llama.num_attention_heads
                    );
                }
                Some(llama.num_attention_heads / group)
            };
            let intermediate = if flag(ffn, "no_op") {
                None
            } else {
                let Some(width) = deci_intermediate(ffn, llama.hidden_size) else {
                    bail!("{} gives layer {layer}'s ffn neither ffn_mult nor intermediate_size", path.display());
                };
                Some(width)
            };
            Ok(LayerPlan { key_value_heads, intermediate })
        })
        .collect()
}
