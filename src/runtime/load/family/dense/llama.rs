//! Llama, Mistral, Mixtral, Phi, PhiMoE, StableLM, StarCoder2, Cohere and Nemotron.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, FeedForwardKind, Names, NormKind, QkvLayout, QueryKeyNorm, Scoring,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::custom::llama4::llama4_temperature;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, number, whole};

/// What the families this file reads add to the Llama block; [`super::super::family`]
/// sends only their `model_type` values here.
pub(in crate::runtime::load::family) fn configure(
    at: &At,
    architecture: &mut Architecture,
) -> Result<()> {
    let At {
        model_type,
        raw,
        scaling,
        layers,
        path,
        ..
    } = *at;
    match model_type {
        "ministral3" => {
            // Ministral 3 (the text decoder of `mistral3`): Mistral's block
            // with Llama 4's query temperature, `llama_4_scaling_beta` past
            // every `original_max_position_embeddings` positions, both
            // stated beside its rotation (vLLM's `llama_4_scaling`).
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.sliding_window = whole(raw, "sliding_window");
            if architecture.sliding_window.is_some() {
                architecture.sliding_layers = every_layer(layers, path)?;
            }
            architecture.query_temperature = llama4_temperature(raw, scaling, layers, path)?;
        }
        "llama" | "mistral" | "mixtral" | "phi3" | "phimoe" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            // Mistral's own weights rotate adjacent pairs (their reading
            // states `rope_interleave`).
            architecture.interleaved_rotary = flag(raw, "rope_interleave");
            if model_type == "phi3" {
                architecture.qkv_layout = QkvLayout::Stacked;
            }
            architecture.fused_feed_forward = model_type == "phi3";
            // Mistral v0.2 and later publish `sliding_window: null`, which is
            // full attention on every layer; Phi-3 states a window it applies
            // on every layer.
            if model_type != "llama" {
                architecture.sliding_window = whole(raw, "sliding_window");
                if architecture.sliding_window.is_some() {
                    architecture.sliding_layers = every_layer(layers, path)?;
                }
            }
            if model_type == "mixtral" {
                architecture.experts = Some(experts(
                    raw,
                    "num_local_experts",
                    "intermediate_size",
                    true,
                    ExpertLayout::Mixtral,
                    0,
                    path,
                )?);
            }
            if model_type == "phimoe" {
                // PhiMoE: Mixtral's tensors under LayerNorms with bias, a
                // biased vocabulary projection when `lm_head_bias` says so,
                // and SparseMixer routing, which Transformers defines for
                // two experts per token.
                architecture.norm = NormKind::Layer { bias: true };
                architecture.lm_head_bias = flag(raw, "lm_head_bias");
                let mut routed = experts(
                    raw,
                    "num_local_experts",
                    "intermediate_size",
                    false,
                    ExpertLayout::Mixtral,
                    0,
                    path,
                )?;
                if routed.top_k != SPARSE_MIXER_EXPERTS {
                    bail!(
                        "{} routes {} experts per token; PhiMoE's SparseMixer chooses exactly {SPARSE_MIXER_EXPERTS}",
                        path.display(),
                        routed.top_k
                    );
                }
                let Some(jitter) = number(raw, "router_jitter_noise") else {
                    bail!(
                        "{} states no router_jitter_noise; PhiMoE's SparseMixer needs it to bound the experts it weighs",
                        path.display()
                    );
                };
                routed.scoring = Scoring::SparseMixer {
                    jitter: jitter as f32,
                };
                architecture.experts = Some(routed);
            }
        }
        "stablelm" => {
            architecture.norm = NormKind::Layer { bias: true };
            architecture.query_key_value_bias = flag(raw, "use_qkv_bias");
            architecture.parallel = flag(raw, "use_parallel_residual");
            if flag(raw, "qk_layernorm") {
                architecture.query_key_norm = QueryKeyNorm::HeadModules;
            }
        }
        "starcoder2" => {
            let bias = raw.get("use_bias").and_then(Value::as_bool).unwrap_or(true);
            architecture.norm = NormKind::Layer { bias: true };
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::STARCODER2;
            architecture.sliding_window = whole(raw, "sliding_window");
            if architecture.sliding_window.is_some() {
                architecture.sliding_layers = every_layer(layers, path)?;
            }
        }
        "cohere" | "cohere2" => {
            architecture.norm = NormKind::Layer { bias: false };
            architecture.parallel = true;
            architecture.interleaved_rotary = true;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.logits_multiplier = number(raw, "logit_scale");
            if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::HeadWeights;
            }
            if model_type == "cohere2" {
                // Every `sliding_window_pattern`-th layer attends globally and
                // applies no rotary embedding; the rest are local and rotate.
                architecture.sliding_window = whole(raw, "sliding_window");
                if let Some(pattern) = whole(raw, "sliding_window_pattern").filter(|p| *p > 0) {
                    fits(layers, path)?;
                    architecture.sliding_layers = (0..layers)
                        .filter(|layer| (layer + 1) % pattern != 0)
                        .fold(0, |set, layer| set | (1u128 << layer));
                }
            }
        }
        "nemotron" => {
            // LayerNorm1P: a LayerNorm whose stored scale is an offset from one.
            architecture.norm = NormKind::Layer { bias: true };
            architecture.norm_offset = true;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::UP_DOWN;
        }
        "phi" => {
            if flag(raw, "qk_layernorm") {
                bail!(
                    "{} declares qk_layernorm; Ster implements Phi without query and key norms",
                    path.display()
                );
            }
            architecture.norm = NormKind::Layer { bias: true };
            architecture.parallel = true;
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.lm_head_bias = true;
            architecture.names = Names::PHI;
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// Transformers' `sparsemixer` (`models/phimoe/modeling_phimoe.py`) runs
/// exactly two selection rounds, so PhiMoE routes two experts per token.
pub(in crate::runtime::load::family) const SPARSE_MIXER_EXPERTS: usize = 2;
