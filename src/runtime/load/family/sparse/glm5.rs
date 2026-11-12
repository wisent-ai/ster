//! GLM-5-Next.

use std::path::Path;

use serde_json::Value;
use {crate::failures::model::Stated, anyhow::Result};

use crate::model::{
    Architecture, DeltaRuleForm, DeltaRuleSpec, DenseClamp, ExpertGroups, ExpertLayout, Scoring,
    SharedExpert, SharedForm, SwigluLimit,
};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::experts::index::sparse_index;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, list, number, stated, whole, whole_or};
use crate::runtime::load::family::sparse::hy_v4::hyper_connections;

/// GLM-5-Next (`glm5_next_text`, inside `glm5_next` below
/// `model.language_model`): hyper-connected residual streams averaged at the
/// end, the layers `layer_types` marks `linear_attention` running Kimi Delta
/// Attention (`linear_attn_config`, its log-decay bounded below by
/// `gate_lower_bound` under `safe_gate`) and the `deepseek_sparse_attention`
/// ones latent attention with no rotation behind a pooled indexer, and the
/// feed-forwards `mlp_layer_types` marks `sparse` routing DeepSeek-V3's way
/// beside shared experts; every SwiGLU, dense, routed or shared, is clamped
/// by `swiglu_limit`.
pub(in crate::runtime::load::family) fn glm5_next(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    if architecture
        .latent
        .is_none_or(|latent| latent.rotated != 0 || latent.query_rank.is_none())
    {
        refused!(
            Unimplemented,
            in raw,
            ["kv_lora_rank", "q_lora_rank", "qk_rope_head_dim"],
            "{} declares GLM-5-Next attention other than latent attention with q_lora_rank and qk_rope_head_dim 0; Ster implements that one",
            path.display()
        );
    }
    let listed = |key: &str, chosen: &str, other: &str| -> Result<u128> {
        let Some(entries) = list(raw, key, path)?
            .filter(|entries| entries.len() == layers)
        else {
            refused!(
                Incomplete,
                in raw,
                [key],
                "{} lists no {key} for its {layers} layers",
                path.display()
            );
        };
        let mut set = 0u128;
        for (layer, entry) in entries.iter().enumerate() {
            match entry.as_str() {
                Some(kind) if kind == chosen => set |= 1u128 << layer,
                Some(kind) if kind == other => {}
                found => refused!(
                    Malformed,
                    in raw,
                    [key],
                    "{} declares {key} entry {layer} as {found:?}; GLM-5-Next's are {chosen} and {other}",
                    path.display()
                ),
            }
        }
        Ok(set)
    };
    let linear = listed(
        "layer_types",
        "linear_attention",
        "deepseek_sparse_attention",
    )?;
    let routed_layers = listed("mlp_layer_types", "sparse", "dense")?;
    const LINEAR: &str = "linear_attn_config";
    if stated::<&serde_json::Map<String, Value>>(raw, LINEAR, path)?.is_none() {
        refused!(
            Incomplete,
            in raw,
            [LINEAR],
            "{} declares a GLM-5-Next model without linear_attn_config",
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
    let (heads, head_dim) = (size("num_heads")?, size("head_dim")?);
    let safe = flag(raw, "linear_attn_config.safe_gate", path)?;
    architecture.delta_rule = Some(DeltaRuleSpec {
        key_heads: heads,
        value_heads: heads,
        key_dim: head_dim,
        value_dim: head_dim,
        kernel: size("short_conv_kernel_size")?,
        layers: linear,
        form: DeltaRuleForm::Kimi,
        negative_eigenvalues: false,
        decay_floor: if safe {
            number(raw, "linear_attn_config.gate_lower_bound", path)?
        } else {
            None
        },
        full_rank_gate: false,
    });
    architecture.sparse_index = Some(sparse_index(
        raw,
        "glm5_next_text",
        layers,
        architecture.latent,
        path,
    )?);
    architecture.hyper_connections = Some(hyper_connections(raw, false, path)?);
    let limit = number(raw, "swiglu_limit", path)?;
    architecture.dense_swiglu_limit = limit.map(DenseClamp::Inner);
    let every = every_layer(layers, path)?;
    let mut routed = experts(
        raw,
        "n_routed_experts",
        "moe_intermediate_size",
        flag(raw, "norm_topk_prob", path)?,
        ExpertLayout::Qwen,
        every & !routed_layers,
        path,
    )?;
    routed.scoring = Scoring::Sigmoid;
    routed.selection_bias = Some("mlp.gate.e_score_correction_bias");
    routed.routed_scale = number(raw, "routed_scaling_factor", path)?;
    routed.swiglu_limit = limit.map(SwigluLimit::Inner);
    let groups = whole_or(raw, "n_group", 1, path)?.max(1);
    let chosen_groups = whole_or(raw, "topk_group", groups, path)?;
    if routed.count % groups != 0 || chosen_groups > groups {
        refused!(
            Contradictory,
            in raw,
            ["n_routed_experts", "n_group", "topk_group"],
            "{} splits {} experts into {groups} groups and keeps {chosen_groups}",
            path.display(),
            routed.count
        );
    }
    routed.groups = Some(ExpertGroups {
        groups,
        chosen_groups,
        rank_by_top_two: true,
    });
    routed.shared = whole(raw, "n_shared_experts", path)?
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    architecture.experts = Some(routed);
    Ok(())
}
