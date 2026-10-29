//! Qwen2, Qwen3, their mixtures of experts, Qwen3-Next, Qwen3.5, MiMo and Mellum.

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, DeltaRuleForm, DeltaRuleSpec, ExpertLayout, QueryKeyNorm, SharedExpert,
    SharedForm,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::{experts, qwen_dense_layers};
use crate::runtime::load::family::read::layers::{every_layer, fits};
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
        layers,
        path,
        ..
    } = *at;
    match model_type {
        "qwen2" | "qwen3" | "qwen2_moe" | "qwen3_moe" | "qwen3_next" | "mimo" | "mellum"
        | "qwen3_5_text" | "qwen3_5_moe_text" => {
            // MiMo is Qwen2 with next-token-prediction layers
            // (`model.mtp_layers`) that a single forward never reads.
            if model_type.starts_with("qwen2") || model_type == "mimo" {
                architecture.query_key_value_bias = true;
            } else {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.query_key_value_bias = flag(raw, "attention_bias");
                architecture.output_bias = architecture.query_key_value_bias;
            }
            if flag(raw, "use_sliding_window") {
                architecture.sliding_window = whole(raw, "sliding_window");
                let from = whole(raw, "max_window_layers").unwrap_or(0);
                architecture.sliding_layers =
                    every_layer(layers, path)? & !every_layer(from.min(layers), path)?;
            }
            // Qwen3.5 (`qwen3_5_text`, `qwen3_5_moe_text`) is Qwen3-Next with
            // its linear attention's projections stored apart.
            let qwen35 = model_type.starts_with("qwen3_5");
            if model_type == "qwen3_next" || qwen35 {
                // Qwen3-Next: norms that store their scale as an offset from
                // one, a sigmoid gate on attention's output from `q_proj`
                // (under `attn_output_gate`, on by default), and gated
                // delta-rule linear attention on every layer `layer_types`
                // calls `linear_attention` (without it, on every layer but
                // each `full_attention_interval`-th).
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
                                    "{} names layer {layer} {other:?}; a Qwen3-Next layer is linear_attention or full_attention",
                                    path.display()
                                ),
                            }
                        }
                        linear
                    }
                    None => {
                        let Some(interval) =
                            whole(raw, "full_attention_interval").filter(|n| *n > 0)
                        else {
                            bail!(
                                "{} declares a Qwen3-Next model with neither layer_types nor full_attention_interval",
                                path.display()
                            );
                        };
                        (0..layers)
                            .filter(|layer| (layer + 1) % interval != 0)
                            .fold(0u128, |set, layer| set | (1u128 << layer))
                    }
                };
                let size = |key: &str| -> Result<usize> {
                    whole(raw, key).filter(|size| *size > 0).with_context(|| {
                        format!(
                            "{} declares a Qwen3-Next model without {key}",
                            path.display()
                        )
                    })
                };
                architecture.norm_offset = true;
                architecture.output_gate = raw
                    .get("attn_output_gate")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                architecture.delta_rule = Some(DeltaRuleSpec {
                    key_heads: size("linear_num_key_heads")?,
                    value_heads: size("linear_num_value_heads")?,
                    key_dim: size("linear_key_head_dim")?,
                    value_dim: size("linear_value_head_dim")?,
                    kernel: size("linear_conv_kernel_dim")?,
                    layers: linear,
                    form: if qwen35 {
                        DeltaRuleForm::Qwen35
                    } else {
                        DeltaRuleForm::Qwen3Next
                    },
                    negative_eigenvalues: false,
                    decay_floor: None,
                    full_rank_gate: false,
                });
            }
            // Mellum is Qwen3-MoE with sliding-window layers, a rotation per
            // layer kind, and dense layers where `mlp_layer_types` says so.
            if model_type.ends_with("_moe")
                || model_type.ends_with("_moe_text")
                || model_type == "qwen3_next"
                || model_type == "mellum"
            {
                // Qwen3-Next and Qwen3.5 renormalise unless told not to
                // (vLLM's `getattr(config, "norm_topk_prob", True)`); Qwen2-
                // and Qwen3-MoE only when told to.
                let normalize = raw
                    .get("norm_topk_prob")
                    .and_then(Value::as_bool)
                    .unwrap_or(model_type == "qwen3_next" || qwen35);
                let mut routed = experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    normalize,
                    ExpertLayout::Qwen,
                    qwen_dense_layers(raw, layers, path)?,
                    path,
                )?;
                if model_type != "qwen3_moe" {
                    routed.shared =
                        whole(raw, "shared_expert_intermediate_size").map(|intermediate| {
                            SharedExpert {
                                intermediate,
                                module: "mlp.shared_expert",
                                gated: true,
                                form: SharedForm::GateUpDown,
                            }
                        });
                }
                architecture.experts = Some(routed);
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
