//! MPT, ChatGLM, GLM, GLM-4, InternLM2, EXAONE 3 and TeleChat2.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    ALIBI_SPAN, Activation, Architecture, FeedForwardKind, Names, NormKind, Positions, QkvLayout,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::read::{flag, number, text};

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
            architecture.query_key_value_bias = flag(raw, "bias");
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
            if flag(raw, "apply_residual_connection_post_layernorm") || flag(raw, "embed_layernorm")
            {
                bail!(
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
            let attention = raw.get("attn_config").cloned().unwrap_or(Value::Null);
            let unsupported = [
                ("qk_ln", flag(&attention, "qk_ln")),
                (
                    "clip_qkv",
                    attention.get("clip_qkv").is_some_and(|v| !v.is_null()),
                ),
                (
                    "softmax_scale",
                    attention.get("softmax_scale").is_some_and(|v| !v.is_null()),
                ),
                (
                    "attn_type",
                    text(&attention, "attn_type").is_some_and(|kind| kind != "multihead_attention"),
                ),
                (
                    "alibi_bias_max",
                    number(&attention, "alibi_bias_max").is_some_and(|max| max != ALIBI_SPAN),
                ),
            ];
            if let Some((key, _)) = unsupported.iter().find(|(_, set)| *set) {
                bail!(
                    "{} declares attn_config.{key} other than MPT's defaults; Ster implements MPT's plain multi-head attention with ALiBi's standard slopes",
                    path.display()
                );
            }
            let alibi = flag(&attention, "alibi");
            if alibi && !llama.num_attention_heads.is_power_of_two() {
                bail!(
                    "{} uses ALiBi over {} heads; Ster implements MPT's slopes for a power-of-two head count",
                    path.display(),
                    llama.num_attention_heads
                );
            }
            let bias = !raw.get("no_bias").and_then(Value::as_bool).unwrap_or(true);
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
            if flag(raw, "apply_residual_connection_post_layernorm") {
                bail!(
                    "{} adds its residual after the norm (apply_residual_connection_post_layernorm); Ster implements ChatGLM's pre-norm block",
                    path.display()
                );
            }
            if raw.get("post_layer_norm").and_then(Value::as_bool) == Some(false) {
                bail!(
                    "{} declares no final norm (post_layer_norm false); Ster implements ChatGLM with its final_layernorm",
                    path.display()
                );
            }
            architecture.names = Names::CHATGLM;
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.fused_feed_forward = true;
            let bias = flag(raw, "add_bias_linear");
            architecture.query_key_value_bias = bias || flag(raw, "add_qkv_bias");
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.interleaved_rotary = raw
                .get("original_rope")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            if raw.get("rmsnorm").and_then(Value::as_bool) == Some(false) {
                architecture.norm = NormKind::Layer { bias: true };
            }
        }
        "glm" | "glm4" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.fused_feed_forward = true;
            architecture.interleaved_rotary = true;
            if model_type == "glm4" {
                architecture.output_norms = true;
                architecture.names = Names::GLM4;
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
