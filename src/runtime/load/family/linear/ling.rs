//! Ling (Bailing), Param2-MoE and Sarvam-30B, and Ling's hybrid forms.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{
    Architecture, ExpertGroups, ExpertLayout, LightningForm, LightningSpec, Names, QkvLayout,
    QueryKeyNorm, Scoring, SharedExpert, SharedForm,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::experts::step::step_limits;
use crate::runtime::load::family::linear::kda::ling_kda;
use crate::runtime::load::family::read::layers::fits;
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
        // Param2-MoE and Sarvam-30B (`sarvam_moe`) are Ling 2.0's block and
        // router under their own names; Ling 2.5 and 3.0 (`bailing_hybrid`)
        // add lightning attention.
        "bailing_moe" | "param2moe" | "bailing_hybrid" | "sarvam_moe" => {
            // Ling 1.x and 2.0: Llama's block under Bailing's names, one
            // stacked `query_key_value`, per-head query and key norms when
            // `use_qk_norm` holds, the first `first_k_dense_replace` layers
            // dense and the rest routed over `num_experts` experts
            // (`mlp.gate`, `mlp.experts.{e}`) beside
            // `moe_intermediate_size · num_shared_experts` shared ones. Ling
            // 1.x scores by softmax and renormalises under `norm_topk_prob`;
            // Ling 2.0's `score_function` sigmoid adds `mlp.gate.expert_bias`
            // to choose (`moe_router_enable_expert_bias`) within its best
            // `topk_group` of `n_group` groups, always renormalises and
            // scales by `routed_scaling_factor`.
            if flag(raw, "norm_head") || flag(raw, "norm_softmax") {
                bail!(
                    "{} normalises its output head (norm_head) or its softmax (norm_softmax); Ster implements Ling's plain lm_head",
                    path.display()
                );
            }
            architecture.names = Names::BAILING;
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.query_key_value_bias = flag(raw, "use_qkv_bias");
            architecture.output_bias = flag(raw, "use_bias");
            if model_type == "bailing_hybrid" {
                bailing_hybrid(raw, layers, llama, architecture, path)?;
            } else if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
            }
            fits(layers, path)?;
            let first_dense = whole(raw, "first_k_dense_replace").unwrap_or(0).min(layers);
            let dense = (0..first_dense).fold(0u128, |set, layer| set | (1u128 << layer));
            let sigmoid = match text(raw, "score_function") {
                None | Some("softmax") => false,
                Some("sigmoid") => true,
                Some(other) => bail!(
                    "{} declares score_function {other:?}; Ster implements softmax and sigmoid expert scores",
                    path.display()
                ),
            };
            let mut routed = experts(
                raw,
                "num_experts",
                "moe_intermediate_size",
                false,
                ExpertLayout::Qwen,
                dense,
                path,
            )?;
            routed.normalize = (sigmoid || flag(raw, "norm_topk_prob")) && routed.top_k > 1;
            routed.shared = whole(raw, "num_shared_experts")
                .filter(|shared| *shared > 0)
                .map(|shared| SharedExpert {
                    intermediate: shared * routed.intermediate,
                    module: "mlp.shared_experts",
                    gated: false,
                    form: SharedForm::GateUpDown,
                });
            if sigmoid {
                routed.scoring = Scoring::Sigmoid;
                routed.routed_scale = number(raw, "routed_scaling_factor");
                routed.selection_bias =
                    flag(raw, "moe_router_enable_expert_bias").then_some("mlp.gate.expert_bias");
                routed.groups = match (whole(raw, "n_group"), whole(raw, "topk_group")) {
                    (Some(groups), Some(chosen_groups)) => {
                        if groups == 0 || routed.count % groups != 0 || chosen_groups > groups {
                            bail!(
                                "{} splits {} experts into {groups} groups and keeps {chosen_groups}; the groups must divide the experts evenly and at least as many must exist as are kept",
                                path.display(),
                                routed.count
                            );
                        }
                        Some(ExpertGroups {
                            groups,
                            chosen_groups,
                            rank_by_top_two: true,
                        })
                    }
                    _ => None,
                };
            }
            if model_type == "bailing_hybrid" {
                step_limits(
                    raw,
                    ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"),
                    layers,
                    &mut routed,
                    path,
                )?;
            }
            architecture.experts = Some(routed);
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// Ling 2.5's and 3.0's additions to Ling 2.0: DeepSeek's latent attention
/// (rotating adjacent pairs under `rope_interleave`) on every
/// `layer_group_size`-th layer, and linear attention on the others. Ling
/// 2.5's is lightning attention, its heads `head_dim` wide with their first
/// `partial_rotary_factor` share rotated by halves at the latent attention's
/// base; Ling 3.0's (`BailingMoeV3ForCausalLM`, or any config stating
/// `short_conv_kernel_size`) is Kimi Delta Attention, see [`ling_kda`].
pub(in crate::runtime::load::family) fn bailing_hybrid(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    let Some(latent) = architecture.latent else {
        bail!(
            "{} declares a Ling hybrid model without kv_lora_rank",
            path.display()
        );
    };
    let ling3 = raw.get("short_conv_kernel_size").is_some()
        || raw
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|names| {
                names
                    .iter()
                    .any(|name| name.as_str() == Some("BailingMoeV3ForCausalLM"))
            });
    if ling3 {
        return ling_kda(raw, layers, llama, architecture, path);
    }
    let group = whole(raw, "layer_group_size").unwrap_or(1).max(1);
    let linear = (0..layers)
        .filter(|layer| (layer + 1) % group != 0)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let Some(head_dim) = whole(raw, "head_dim") else {
        bail!(
            "{} declares a Ling hybrid model without head_dim",
            path.display()
        );
    };
    let heads = llama.num_attention_heads;
    if whole(raw, "num_kv_heads_for_linear_attn").is_some_and(|stated| stated != heads) {
        bail!(
            "{} gives its lightning attention a key-value head count other than its {heads} heads; Ster runs it with one key and value per head",
            path.display()
        );
    }
    let share = number(raw, "partial_rotary_factor").unwrap_or(1.0);
    let rotated = (head_dim as f64 * share) as usize;
    if rotated != latent.rotated {
        bail!(
            "{} rotates {rotated} components of each lightning head and {} of each latent one; Ster rotates both by one table",
            path.display(),
            latent.rotated
        );
    }
    let groups = whole(raw, "group_norm_size").unwrap_or(1).max(1);
    if (heads * head_dim) % groups != 0 {
        bail!(
            "{} splits its {} lightning channels into {groups} norm groups unevenly",
            path.display(),
            heads * head_dim
        );
    }
    architecture.interleaved_rotary = raw
        .get("rope_interleave")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    architecture.lightning = Some(LightningSpec {
        heads,
        head_dim,
        layers: linear,
        form: LightningForm::Bailing {
            groups,
            silu: flag(raw, "linear_silu") || flag(raw, "use_linear_silu"),
            qk_norm: flag(raw, "use_qk_norm"),
        },
    });
    Ok(())
}
