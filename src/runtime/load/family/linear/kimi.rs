//! Kimi-Linear.

use serde_json::{Map, Value};
use {crate::failures::model::Stated, anyhow::Result};

use crate::model::{
    Architecture, DeltaRuleForm, DeltaRuleSpec, ExpertGroups, ExpertLayout, GateFunction,
    LatentExperts, Positions, Scoring, SharedExpert, SharedForm,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{
    flag, flag_or, list, number, stated, text, whole, whole_or,
};

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
        "kimi_linear" => {
            // Kimi-Linear: Kimi Delta Attention on the layers
            // `linear_attn_config.kda_layers` names (counted from one),
            // multi-head latent attention on the rest (unrotated when
            // `mla_use_nope` holds), and a mixture of `num_experts`
            // Mixtral-named experts under `block_sparse_moe` beside shared
            // experts, routed as DeepSeek-V3 routes.
            fits(layers, path)?;
            if architecture.latent.is_none() {
                refused!(
                    Incomplete,
                    in raw,
                    ["kv_lora_rank"],
                    "{} declares no kv_lora_rank; Ster implements Kimi-Linear's attention layers as latent attention",
                    path.display()
                );
            }
            const LINEAR: &str = "linear_attn_config";
            if stated::<&Map<String, Value>>(raw, LINEAR, path)?.is_none() {
                refused!(
                    Incomplete,
                    in raw,
                    [LINEAR],
                    "{} declares a Kimi-Linear model without linear_attn_config",
                    path.display()
                );
            }
            let size = |key: &str| -> Result<usize> {
                let at = format!("{LINEAR}.{key}");
                whole(raw, &at, path)?
                    .filter(|size| *size > 0)
                    .or_refuse(raw, &[at.as_str()], || {
                        format!("{} declares linear_attn_config without {key}", path.display())
                    })
            };
            let Some(listed) = list(raw, "linear_attn_config.kda_layers", path)? else {
                refused!(
                    Incomplete,
                    in raw,
                    ["linear_attn_config.kda_layers"],
                    "{} declares linear_attn_config without kda_layers",
                    path.display()
                );
            };
            let mut kda = 0u128;
            for entry in listed {
                match entry.as_u64().map(|layer| layer as usize) {
                    Some(layer) if (1..=layers).contains(&layer) => kda |= 1u128 << (layer - 1),
                    _ => refused!(
                        OutOfRange,
                        in raw,
                        ["linear_attn_config.kda_layers"],
                        "{} lists kda_layers entry {entry}, outside layers 1 to {layers}",
                        path.display()
                    ),
                }
            }
            let (heads, head_dim) = (size("num_heads")?, size("head_dim")?);
            architecture.delta_rule = Some(DeltaRuleSpec {
                key_heads: heads,
                value_heads: heads,
                key_dim: head_dim,
                value_dim: head_dim,
                kernel: size("short_conv_kernel_size")?,
                layers: kda,
                form: DeltaRuleForm::Kimi,
                negative_eigenvalues: false,
                decay_floor: number(raw, "linear_attn_config.gate_lower_bound", path)?,
                full_rank_gate: flag_or(raw, "linear_attn_config.use_full_rank_gate", false, path)?,
            });
            if flag(raw, "mla_use_nope", path)? {
                architecture.positions = Positions::None;
            }
            let first_dense = whole_or(raw, "first_k_dense_replace", 0, path)?;
            let frequency = whole_or(raw, "moe_layer_freq", 1, path)?.max(1);
            let dense = (0..layers)
                .filter(|layer| *layer < first_dense || layer % frequency != 0)
                .fold(0u128, |set, layer| set | (1u128 << layer));
            let mut routed = experts(
                raw,
                "num_experts",
                "moe_intermediate_size",
                flag(raw, "moe_renormalize", path)?,
                ExpertLayout::Mixtral,
                dense,
                path,
            )?;
            routed.scoring = match text(raw, "moe_router_activation_func", path)? {
                None | Some("softmax") => Scoring::Softmax,
                Some("sigmoid") => Scoring::Sigmoid,
                Some(other) => refused!(
                    Unimplemented,
                    in raw,
                    ["moe_router_activation_func"],
                    "{} declares moe_router_activation_func {other:?}; Ster implements softmax and sigmoid expert scores",
                    path.display()
                ),
            };
            routed.selection_bias = Some("block_sparse_moe.gate.e_score_correction_bias");
            routed.routed_scale = number(raw, "routed_scaling_factor", path)?;
            if flag(raw, "use_grouped_topk", path)? {
                let groups = whole_or(raw, "num_expert_group", 1, path)?;
                let chosen_groups = whole_or(raw, "topk_group", groups, path)?;
                if groups == 0 || routed.count % groups != 0 || chosen_groups > groups {
                    refused!(
                        Contradictory,
                        in raw,
                        ["num_experts", "num_expert_group", "topk_group"],
                        "{} splits {} experts into {groups} groups and keeps {chosen_groups}; the groups must divide the experts evenly and at least as many must exist as are kept",
                        path.display(),
                        routed.count
                    );
                }
                routed.groups = Some(ExpertGroups {
                    groups,
                    chosen_groups,
                    rank_by_top_two: true,
                });
            }
            routed.shared = whole(raw, "num_shared_experts", path)?
                .filter(|shared| *shared > 0)
                .map(|shared| SharedExpert {
                    intermediate: shared * routed.intermediate,
                    module: "block_sparse_moe.shared_experts",
                    gated: false,
                    form: SharedForm::GateUpDown,
                });
            // Kimi-K3's latent experts: `routed_expert_hidden_size` wide,
            // `routed_expert_norm` over their sum under `latent_moe_use_norm`.
            let latent_norm = flag(raw, "latent_moe_use_norm", path)?
                .then_some((LatentExperts::KIMI_NORM, architecture.norm_eps));
            routed.latent = whole(raw, "routed_expert_hidden_size", path)?
                .filter(|width| *width > 0)
                .map(|width| LatentExperts {
                    width,
                    down: LatentExperts::KIMI_DOWN,
                    up: LatentExperts::KIMI_UP,
                    norm: latent_norm,
                });
            architecture.experts = Some(routed);
            // Kimi-K3's `mla_use_output_gate`: attention's output times the
            // sigmoid of `g_proj`, elementwise, before `o_proj`.
            if flag(raw, "mla_use_output_gate", path)? {
                architecture.attention_gate = Some(GateFunction::Sigmoid);
                architecture.names.attention_gate = "g_proj";
            }
            architecture.depth_block_size = match whole(raw, "attn_res_block_size", path)? {
                Some(0) => refused!(
                    OutOfRange,
                    in raw,
                    ["attn_res_block_size"],
                    "{} declares attn_res_block_size 0; a block holds at least one layer",
                    path.display()
                ),
                size => size,
            };
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
