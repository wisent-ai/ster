//! MPT, ChatGLM, GLM, GLM-4, InternLM2, EXAONE 3 and TeleChat2.

use anyhow::Result;
use serde_json::{Map, Value};

use crate::model::{
    ALIBI_SPAN, Activation, Architecture, FeedForwardKind, Names, NormKind, Positions, QkvLayout,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::read::{flag, flag_or, number, stated, text};

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
        path,
        ..
    } = *at;
    match model_type {
        "internlm2" => {
            // InternLM2: Llama's block under its own names, query, key and
            // value fused in `wqkv` by key-value group, and `bias` on every
            // attention projection.
            architecture.names = Names::INTERNLM2;
            architecture.qkv_layout = QkvLayout::Grouped;
            architecture.query_key_value_bias = flag(raw, "bias", path)?;
            architecture.output_bias = architecture.query_key_value_bias;
        }
        "exaone" => {
            // EXAONE 3 and 3.5: Llama's block under GPT-2-style names.
            architecture.names = Names::EXAONE;
        }
        "telechat" => {
            // TeleChat2: Llama's block below `transformer`, a separate
            // `query` beside a `key_value` matrix paired per head, biases on
            // the attention output and the down projection only.
            if flag(raw, "apply_residual_connection_post_layernorm", path)? || flag(raw, "embed_layernorm", path)?
            {
                refused!(
                    Unimplemented,
                    in raw,
                    ["apply_residual_connection_post_layernorm", "embed_layernorm"],
                    "{} adds its residual after the norm (apply_residual_connection_post_layernorm) or normalises the embeddings (embed_layernorm); Ster implements TeleChat2's pre-norm block without an embedding norm",
                    path.display()
                );
            }
            architecture.names = Names::TELECHAT;
            architecture.qkv_layout = QkvLayout::PairedKeyValue;
            architecture.output_bias = true;
            architecture.down_bias = true;
        }
        "mpt" => {
            // MPT: `transformer.blocks`, LayerNorm and projections without
            // bias under `no_bias`, one `Wqkv`, the exact GELU, and ALiBi or a
            // learned `wpe` as `attn_config` says.
            let attention = stated::<&Map<String, Value>>(raw, "attn_config", path)?;
            let set = |key: &str| {
                attention
                    .and_then(|attention| attention.get(key))
                    .is_some_and(|value| !value.is_null())
            };
            let unsupported = [
                ("qk_ln", flag(raw, "attn_config.qk_ln", path)?),
                ("clip_qkv", set("clip_qkv")),
                ("softmax_scale", set("softmax_scale")),
                (
                    "attn_type",
                    text(raw, "attn_config.attn_type", path)?
                        .is_some_and(|kind| kind != "multihead_attention"),
                ),
                (
                    "alibi_bias_max",
                    number(raw, "attn_config.alibi_bias_max", path)?
                        .is_some_and(|max| max != ALIBI_SPAN),
                ),
            ];
            if let Some((key, _)) = unsupported.iter().find(|(_, set)| *set) {
                refused!(
                    Unimplemented,
                    in raw,
                    [format!("attn_config.{key}")],
                    "{} declares attn_config.{key} other than MPT's defaults; Ster implements MPT's plain multi-head attention with ALiBi's standard slopes",
                    path.display()
                );
            }
            let alibi = flag(raw, "attn_config.alibi", path)?;
            if alibi && !llama.num_attention_heads.is_power_of_two() {
                refused!(
                    Unimplemented,
                    ["num_attention_heads" => llama.num_attention_heads],
                    "{} uses ALiBi over {} heads; Ster implements MPT's slopes for a power-of-two head count",
                    path.display(),
                    llama.num_attention_heads
                );
            }
            let bias = !flag_or(raw, "no_bias", true, path)?;
            architecture.names = Names::MPT;
            architecture.norm = NormKind::Layer { bias };
            architecture.positions = if alibi {
                Positions::Alibi {
                    inside_scale: false,
                }
            } else {
                Positions::Learned { offset: 0 }
            };
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.activation = Activation::Gelu;
        }
        "chatglm" => {
            // ChatGLM (GLM-4-9B and ChatGLM2/3): Llama's block under its own
            // names, query, key and value stacked in `query_key_value`
            // (biased under `add_qkv_bias`), gate and up stacked in
            // `dense_h_to_4h`, the first half of each head rotating —
            // adjacent pairs under `original_rope` — and RMS norms under
            // `rmsnorm`, LayerNorms otherwise.
            if flag(raw, "apply_residual_connection_post_layernorm", path)? {
                refused!(
                    Unimplemented,
                    in raw,
                    ["apply_residual_connection_post_layernorm"],
                    "{} adds its residual after the norm (apply_residual_connection_post_layernorm); Ster implements ChatGLM's pre-norm block",
                    path.display()
                );
            }
            if !flag_or(raw, "post_layer_norm", true, path)? {
                refused!(
                    Unimplemented,
                    in raw,
                    ["post_layer_norm"],
                    "{} declares no final norm (post_layer_norm false); Ster implements ChatGLM with its final_layernorm",
                    path.display()
                );
            }
            architecture.names = Names::CHATGLM;
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.fused_feed_forward = true;
            let bias = flag(raw, "add_bias_linear", path)?;
            architecture.query_key_value_bias = bias || flag(raw, "add_qkv_bias", path)?;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.interleaved_rotary = flag_or(raw, "original_rope", true, path)?;
            if !flag_or(raw, "rmsnorm", true, path)? {
                architecture.norm = NormKind::Layer { bias: true };
            }
        }
        "glm" | "glm4" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias", path)?;
            architecture.fused_feed_forward = true;
            architecture.interleaved_rotary = true;
            if model_type == "glm4" {
                architecture.output_norms = true;
                architecture.names = Names::GLM4;
            }
        }
        other => refused!(
            Unimplemented,
            in raw,
            ["model_type"],
            "model architecture {other:?} has no decoder in this Ster build"
        ),
    }
    Ok(())
}
