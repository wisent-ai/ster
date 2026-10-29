//! OLMo 1, 2 and 3, OLMoE, FlexOlmo, OLMo Hybrid, SmolLM3, DBRX, EXAONE 4 and Apertus.

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, DeltaRuleForm, DeltaRuleSpec, ExpertLayout, FeedForwardKind, Names, NormKind,
    Positions, QkvLayout, QueryKeyNorm,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, fits, unrotated};
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
        layers,
        path,
        ..
    } = *at;
    match model_type {
        "olmoe" => {
            architecture.clip_qkv = number(raw, "clip_qkv");
            architecture.query_key_norm = QueryKeyNorm::Full;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.experts = Some(experts(
                raw,
                "num_experts",
                "intermediate_size",
                flag(raw, "norm_topk_prob"),
                ExpertLayout::Qwen,
                0,
                path,
            )?);
        }
        "olmo_hybrid" => {
            // OLMo Hybrid: Gated DeltaNet (`linear_attn`) in a pre-norm block
            // (`input_layernorm`, the mixer, `post_attention_layernorm`, the
            // feed-forward) on the layers `layer_types` calls
            // `linear_attention` — each fourth from the fourth attending
            // when it lists none — and OLMo 3's post-norm attention block
            // with whole-projection query and key norms on the rest, rotated
            // only when `rope_parameters` states a `rope_theta`.
            fits(layers, path)?;
            let linear = match raw.get("layer_types").and_then(Value::as_array) {
                Some(types) => {
                    if types.len() != layers {
                        bail!(
                            "{} lists {} layer_types for {layers} layers",
                            path.display(),
                            types.len()
                        );
                    }
                    let mut linear = 0u128;
                    for (layer, kind) in types.iter().enumerate() {
                        match kind.as_str() {
                            Some("linear_attention") => linear |= 1u128 << layer,
                            Some("full_attention") => {}
                            other => bail!(
                                "{} names layer {layer} {other:?}; an OLMo Hybrid layer is linear_attention or full_attention",
                                path.display()
                            ),
                        }
                    }
                    linear
                }
                None => {
                    let attending =
                        |layer: usize| layer % OLMO_HYBRID_FULL_EVERY == OLMO_HYBRID_FULL_EVERY - 1;
                    let mut linear = (0..layers)
                        .filter(|layer| !attending(*layer))
                        .fold(0u128, |set, layer| set | (1u128 << layer));
                    if layers > 0 && (0..layers).all(|layer| !attending(layer)) {
                        linear &= !(1u128 << (layers - 1));
                    }
                    linear
                }
            };
            let size = |key: &str| -> Result<usize> {
                whole(raw, key).filter(|size| *size > 0).with_context(|| {
                    format!(
                        "{} declares an OLMo Hybrid model without {key}",
                        path.display()
                    )
                })
            };
            architecture.query_key_norm = QueryKeyNorm::Full;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.pre_norms = false;
            architecture.output_norms = true;
            architecture.delta_rule = Some(DeltaRuleSpec {
                key_heads: size("linear_num_key_heads")?,
                value_heads: size("linear_num_value_heads")?,
                key_dim: size("linear_key_head_dim")?,
                value_dim: size("linear_value_head_dim")?,
                kernel: size("linear_conv_kernel_dim")?,
                layers: linear,
                form: DeltaRuleForm::OlmoHybrid,
                negative_eigenvalues: raw
                    .get("linear_allow_neg_eigval")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                decay_floor: None,
                full_rank_gate: false,
            });
            let stated_theta = |object: Option<&Value>| {
                object
                    .and_then(|value| value.get("rope_theta"))
                    .is_some_and(Value::is_number)
            };
            if !stated_theta(raw.get("rope_parameters")) && !stated_theta(Some(raw)) {
                architecture.positions = Positions::None;
            }
        }
        "olmo2" | "olmo3" | "flex_olmo" => {
            architecture.query_key_norm = QueryKeyNorm::Full;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.pre_norms = false;
            architecture.output_norms = true;
            if model_type == "olmo3" {
                architecture.sliding_window = whole(raw, "sliding_window");
            }
            // FlexOlmo: OLMo 2's block with OLMoE's experts as its
            // feed-forward.
            if model_type == "flex_olmo" {
                architecture.experts = Some(experts(
                    raw,
                    "num_experts",
                    "intermediate_size",
                    flag(raw, "norm_topk_prob"),
                    ExpertLayout::Qwen,
                    0,
                    path,
                )?);
            }
        }
        "smollm3" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            if let Some(flags) = raw.get("no_rope_layers").and_then(Value::as_array) {
                architecture.unrotated_layers = unrotated(flags, layers, path)?;
            }
            if flag(raw, "use_sliding_window") {
                architecture.sliding_window = whole(raw, "sliding_window");
                architecture.sliding_layers = every_layer(layers, path)?;
            }
        }
        "olmo" => {
            // OLMo 1's LayerNorm stores neither a scale nor a bias.
            architecture.norm = NormKind::Bare;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.clip_qkv = number(raw, "clip_qkv");
        }
        "dbrx" => {
            // DBRX: LayerNorms without bias, one `Wqkv` clipped to
            // `clip_qkv`, and a mixture of `moe_num_experts` experts whose
            // top `moe_top_k` weights are divided by their p-norm, p being
            // `moe_normalize_expert_weights`.
            let normalize = match raw.get("moe_normalize_expert_weights") {
                None | Some(Value::Null) => false,
                Some(power) if power.as_f64() == Some(1.0) => true,
                Some(other) => bail!(
                    "{} normalises expert weights by their {other}-norm; Ster implements the 1-norm (moe_normalize_expert_weights 1) or none",
                    path.display()
                ),
            };
            architecture.names = Names::DBRX;
            architecture.norm = NormKind::Layer { bias: false };
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.clip_qkv = number(raw, "clip_qkv");
            architecture.experts = Some(experts(
                raw,
                "moe_num_experts",
                "intermediate_size",
                normalize,
                ExpertLayout::Dbrx,
                0,
                path,
            )?);
        }
        "exaone4" => {
            // OLMo 2's post-norm block with per-head query and key norms.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.pre_norms = false;
            architecture.output_norms = true;
            architecture.sliding_window = whole(raw, "sliding_window");
            if let Some(pattern) = whole(raw, "sliding_window_pattern").filter(|p| *p > 0) {
                fits(layers, path)?;
                architecture.sliding_layers = (0..layers)
                    .filter(|layer| (layer + 1) % pattern != 0)
                    .fold(0, |set, layer| set | (1u128 << layer));
            }
        }
        "apertus" => {
            // Apertus: per-head query and key norms, `attention_layernorm`
            // and `feedforward_layernorm`, and a plain up-activation-down
            // feed-forward whose xIELU keeps its parameters in `mlp.act_fn`.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::APERTUS;
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// How often OLMo Hybrid's default layout attends: each fourth layer from
/// the fourth (`OlmoHybridConfig.__post_init__`, `i % 4 == 3`).
pub(in crate::runtime::load::family) const OLMO_HYBRID_FULL_EVERY: usize = 4;
