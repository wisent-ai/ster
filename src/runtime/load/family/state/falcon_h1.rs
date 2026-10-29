//! Falcon-H1 and LFM2.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, Names, ParallelScan, ParameterNorm, QueryKeyNorm, Scoring,
    ShortConvolution, StateSpaceSpec,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, number, numbers, whole};
use crate::runtime::load::family::state::mamba::structured;

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
        "falcon_h1" => {
            // Falcon-H1: every layer runs attention and a Mamba-2 scan side
            // by side on one normed input (`input_layernorm`), then
            // `pre_ff_layernorm` and the feed-forward, all under Jamba's
            // names, with muP multipliers on the embeddings, the attention
            // input and output, every key, the scan input, the five parts
            // of its projection and its output, the feed-forward's gate and
            // output, and the logits.
            let scale = |key: &str| number(raw, key).unwrap_or(1.0);
            if raw
                .get("attn_layer_indices")
                .is_some_and(|indices| !indices.is_null())
                || raw.get("mamba_use_mlp").and_then(Value::as_bool) == Some(false)
                || flag(raw, "mamba_norm_before_gate")
                || flag(raw, "mamba_proj_bias") != flag(raw, "projectors_bias")
            {
                bail!(
                    "{} declares attn_layer_indices, mamba_use_mlp false, mamba_norm_before_gate, or a projectors_bias unlike mamba_proj_bias; Ster implements Falcon-H1 with attention, a scan and a feed-forward on every layer, the gate before the norm, and one bias setting for the scan's projections",
                    path.display()
                );
            }
            let mut heads =
                structured(raw, "mamba_n_heads", "mamba_d_head", "mamba_n_groups", path)?;
            let (Some(state), Some(kernel)) =
                (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
            else {
                bail!(
                    "{} declares a Falcon-H1 model without mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            let inner = whole(raw, "mamba_d_ssm").unwrap_or_else(|| {
                (number(raw, "mamba_expand").unwrap_or(2.0) * llama.hidden_size as f64) as usize
            });
            if inner != heads.heads * heads.head_dim {
                bail!(
                    "{} sizes its scan at {inner} but {} heads of {} make {}; the inner width must be the heads' total",
                    path.display(),
                    heads.heads,
                    heads.head_dim,
                    heads.heads * heads.head_dim
                );
            }
            let [gate_scale, output_scale] = numbers::<2>(raw, "mlp_multipliers", path)?;
            heads.input_scale = scale("ssm_in_multiplier");
            heads.projection_scales = Some(numbers::<5>(raw, "ssm_multipliers", path)?);
            heads.gated_norm = raw
                .get("mamba_rms_norm")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            architecture.names = Names::JAMBA;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.embedding_multiplier = number(raw, "embedding_multiplier");
            architecture.logits_multiplier = number(raw, "lm_head_multiplier");
            architecture.key_scale = number(raw, "key_multiplier");
            architecture.feed_forward_scales = Some((gate_scale, output_scale));
            architecture.parallel_scan = Some(ParallelScan {
                attention_in: scale("attention_in_multiplier"),
                attention_out: scale("attention_out_multiplier"),
                scan_out: scale("ssm_out_multiplier"),
            });
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "mamba_proj_bias"),
                convolution_bias: raw
                    .get("mamba_conv_bias")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                // The scan runs inside every layer's parallel block, never
                // as a layer of its own.
                layers: 0,
                feed_forward: true,
                structured: Some(heads),
            });
        }
        "lfm2" | "lfm2_moe" => {
            // LFM2: gated short convolutions on every layer but those
            // `full_attn_idxs` (or `layer_types`' `full_attention`) names,
            // per-head query and key norms on the attention layers.
            fits(layers, path)?;
            let attention: u128 = match raw.get("layer_types").and_then(Value::as_array) {
                Some(types) => types
                    .iter()
                    .enumerate()
                    .filter(|(_, kind)| kind.as_str() == Some("full_attention"))
                    .fold(0, |set, (layer, _)| set | (1u128 << layer)),
                None => raw
                    .get("full_attn_idxs")
                    .and_then(Value::as_array)
                    .map(|list| {
                        list.iter()
                            .filter_map(Value::as_u64)
                            .filter(|layer| (*layer as usize) < layers)
                            .fold(0u128, |set, layer| set | (1u128 << layer))
                    })
                    .unwrap_or(0),
            };
            let Some(kernel) = whole(raw, "conv_L_cache") else {
                bail!(
                    "{} declares an LFM2 model without conv_L_cache",
                    path.display()
                );
            };
            architecture.names = Names::LFM2;
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.short_convolution = Some(ShortConvolution {
                kernel,
                bias: flag(raw, "conv_bias"),
                layers: every_layer(layers, path)? & !attention,
            });
            if model_type == "lfm2_moe" {
                // LFM2-MoE: the first `num_dense_layers` layers keep the
                // plain `intermediate_size` feed-forward; the rest route
                // over `num_experts` experts (`feed_forward.gate`,
                // `feed_forward.experts.{e}.w1`/`w3`/`w2`) by sigmoid
                // scores, chosen with `feed_forward.expert_bias` added when
                // `use_expert_bias` holds, renormalised under
                // `norm_topk_prob`, scaled by `routed_scaling_factor`.
                let dense_count = whole(raw, "num_dense_layers").unwrap_or(0).min(layers);
                let dense = (0..dense_count).fold(0u128, |set, layer| set | (1u128 << layer));
                let mut routed = experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    flag(raw, "norm_topk_prob"),
                    ExpertLayout::Lfm2,
                    dense,
                    path,
                )?;
                routed.scoring = Scoring::Sigmoid;
                routed.selection_bias =
                    flag(raw, "use_expert_bias").then_some("feed_forward.expert_bias");
                routed.routed_scale = number(raw, "routed_scaling_factor");
                architecture.experts = Some(routed);
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
