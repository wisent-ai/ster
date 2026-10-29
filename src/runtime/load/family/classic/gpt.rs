//! GPT-NeoX, GPT-J, GPT-2, GPT-BigCode, OPT, BLOOM and Falcon.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Activation, Architecture, FeedForwardKind, Names, NormKind, Positions, QkvLayout,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::read::{flag, whole};

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
        "gpt_neox" => {
            // Pythia and GPT-NeoX: LayerNorm with bias, a head-interleaved
            // `query_key_value`, biased projections and a plain GELU
            // feed-forward; parallel blocks by default, each half with its own
            // norm.
            architecture.names = Names::GPT_NEOX;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.qkv_layout = QkvLayout::Grouped;
            let bias = raw
                .get("attention_bias")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.parallel = raw
                .get("use_parallel_residual")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            architecture.parallel_norms = true;
        }
        "gptj" => {
            // GPT-J: one `ln_1` before both halves of a parallel block,
            // unbiased attention, a biased plain feed-forward and head, and
            // interleaved rotation over `rotary_dim`.
            architecture.names = Names::GPT_J;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.parallel = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.lm_head_bias = true;
            architecture.interleaved_rotary = true;
        }
        "gpt2" | "gpt_bigcode" => {
            // GPT-2 and GPT-BigCode: a learned position table, LayerNorm with
            // bias, one biased `c_attn` holding query, key and value (one
            // key-value head when GPT-BigCode's `multi_query`), and a biased
            // plain feed-forward. GPT-2 stores every projection `[inputs,
            // outputs]`.
            if flag(raw, "scale_attn_by_inverse_layer_idx") {
                bail!(
                    "{} declares scale_attn_by_inverse_layer_idx; Ster scales every layer's attention by the head width alone",
                    path.display()
                );
            }
            architecture.names = Names::GPT2;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.positions = Positions::Learned { offset: 0 };
            architecture.conv1d = model_type == "gpt2";
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
        }
        "opt" => {
            // OPT: a learned position table read two rows in, pre-norm
            // blocks with biased projections and a plain ReLU feed-forward.
            if raw.get("do_layer_norm_before").and_then(Value::as_bool) == Some(false) {
                bail!(
                    "{} declares do_layer_norm_before false; Ster implements OPT's pre-norm blocks",
                    path.display()
                );
            }
            if let Some(width) =
                whole(raw, "word_embed_proj_dim").filter(|w| *w != llama.hidden_size)
            {
                bail!(
                    "{} projects {width}-wide word embeddings into a {}-wide model (word_embed_proj_dim); Ster implements OPT with embeddings as wide as the model",
                    path.display(),
                    llama.hidden_size
                );
            }
            let bias = raw
                .get("enable_bias")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            architecture.names = Names::OPT;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.positions = Positions::Learned {
                offset: OPT_POSITION_OFFSET,
            };
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
        }
        "bloom" => {
            // BLOOM: ALiBi instead of positions, a norm after the embedding,
            // a head-interleaved biased `query_key_value`, and a biased plain
            // feed-forward with the tanh GELU.
            if flag(raw, "apply_residual_connection_post_layernorm") {
                bail!(
                    "{} declares apply_residual_connection_post_layernorm; Ster implements BLOOM with the residual taken before each norm",
                    path.display()
                );
            }
            architecture.names = Names::BLOOM;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.embedding_norm = true;
            architecture.positions = Positions::Alibi {
                inside_scale: false,
            };
            architecture.qkv_layout = QkvLayout::Grouped;
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.activation = Activation::GeluTanh;
        }
        "falcon" => {
            // Falcon: BLOOM's tensor names, rows grouped by key-value head
            // (one group for the multi-query models, `num_kv_heads` for the
            // new decoder architecture, one per head otherwise), a parallel
            // block with one norm or, in the new architecture, `ln_attn` and
            // `ln_mlp`, the exact GELU, and rotation or ALiBi.
            let new = flag(raw, "new_decoder_architecture");
            let two_norms = new && whole(raw, "num_ln_in_parallel_attn") != Some(1);
            architecture.names = if two_norms {
                Names::FALCON_TWO_NORMS
            } else {
                Names::FALCON
            };
            architecture.parallel = new || flag(raw, "parallel_attn");
            architecture.parallel_norms = two_norms;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.qkv_layout = QkvLayout::Grouped;
            let bias = flag(raw, "bias");
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.activation = Activation::Gelu;
            if flag(raw, "alibi") {
                architecture.positions = Positions::Alibi { inside_scale: true };
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// OPT's learned position table keeps this many rows before position zero;
/// Transformers' `OPTLearnedPositionalEmbedding` reads `position + offset`.
pub(in crate::runtime::load::family) const OPT_POSITION_OFFSET: usize = 2;
