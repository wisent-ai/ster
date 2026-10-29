//! DeepSeek's mixture of experts and router.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    ExpertGroups, ExpertLayout, MixtureOfExperts, Scoring, SharedExpert, SharedForm,
};
use crate::runtime::load::family::experts::{experts, qwen_dense_layers};
use crate::runtime::load::family::read::layers::{every_layer, fits, flagged_layers};
use crate::runtime::load::family::read::{flag, number, text, whole};

/// DeepSeek's mixture of experts: `n_routed_experts` routed and
/// `n_shared_experts` shared experts of `moe_intermediate_size` each, dense
/// layers before `first_k_dense_replace` and off `moe_layer_freq`, scores by
/// `scoring_func`, chosen by `topk_method` (`greedy`, or group-limited over
/// `n_group` groups keeping `topk_group`; `noaux_tc` ranks groups by their
/// best two and adds `e_score_correction_bias` to choose), and weights scaled
/// by `routed_scaling_factor` — always on V3, on V2 only when they are not
/// renormalised.
pub(in crate::runtime::load::family) fn deepseek_experts(
    raw: &Value,
    model_type: &str,
    layers: usize,
    path: &Path,
) -> Result<MixtureOfExperts> {
    fits(layers, path)?;
    let first_dense = whole(raw, "first_k_dense_replace").unwrap_or(0);
    let frequency = whole(raw, "moe_layer_freq").unwrap_or(1).max(1);
    let dense_layers = if raw.get("mlp_layer_types").is_some_and(Value::is_array) {
        qwen_dense_layers(raw, layers, path)?
    } else if raw.get("moe_layer_freq").is_some_and(Value::is_array) {
        // MiMo-V2 lists `moe_layer_freq` per layer: one for routed, zero
        // for dense.
        every_layer(layers, path)? & !flagged_layers(raw, "moe_layer_freq", layers, path)?
    } else {
        (0..layers)
            .filter(|layer| *layer < first_dense || layer % frequency != 0)
            .fold(0, |set, layer| set | (1u128 << layer))
    };
    let normalize = flag(raw, "norm_topk_prob");
    let mut routed = experts(
        raw,
        // K-EXAONE spells the counts `num_experts` and `num_shared_experts`.
        if model_type == "exaone_moe" {
            "num_experts"
        } else {
            "n_routed_experts"
        },
        "moe_intermediate_size",
        normalize,
        ExpertLayout::Qwen,
        dense_layers,
        path,
    )?;
    routed.shared = whole(
        raw,
        if model_type == "exaone_moe" {
            "num_shared_experts"
        } else {
            "n_shared_experts"
        },
    )
    .filter(|shared| *shared > 0)
    .map(|shared| SharedExpert {
        intermediate: shared * routed.intermediate,
        module: "mlp.shared_experts",
        gated: false,
        form: SharedForm::GateUpDown,
    });
    // GLM-4-MoE's and Nemotron-H's routers are DeepSeek-V3's and their
    // configs leave the method out: sigmoid scores, `noaux_tc` selection.
    let v3_default = matches!(
        model_type,
        "glm4_moe" | "glm4_moe_lite" | "glm_moe_dsa" | "exaone_moe"
    ) || model_type.starts_with("nemotron_h");
    let v3_router = v3_default
        || matches!(
            model_type,
            "deepseek_v3" | "deepseek_v32" | "axk1" | "pangu_ultra_moe" | "dots3_note"
        );
    routed.scoring = match text(raw, "scoring_func") {
        None if v3_default => Scoring::Sigmoid,
        None | Some("softmax") => Scoring::Softmax,
        Some("sigmoid") => Scoring::Sigmoid,
        Some(other) => bail!(
            "{} declares scoring_func {other:?}; Ster implements softmax and sigmoid expert scores",
            path.display()
        ),
    };
    let method = text(raw, "topk_method").unwrap_or(if v3_default { "noaux_tc" } else { "greedy" });
    // A.X-K1's `topk_method` `none` is vLLM's grouped top-k with no
    // selection bias, which ranks each group by its best expert: DeepSeek-V2's
    // `group_limited_greedy`.
    let method = if model_type == "axk1" && method == "none" {
        "group_limited_greedy"
    } else {
        method
    };
    routed.groups = match method {
        "greedy" => None,
        // Dots3-Note's configs state no groups; its config class keeps every
        // expert in one group (`Dots3NoteConfig` sets `n_group` and
        // `topk_group` to one), so every expert competes on its score plus
        // the selection bias.
        "noaux_tc"
            if model_type == "dots3_note"
                && whole(raw, "n_group").is_none()
                && whole(raw, "topk_group").is_none() =>
        {
            None
        }
        "group_limited_greedy" | "noaux_tc" => {
            let (Some(groups), Some(chosen_groups)) =
                (whole(raw, "n_group"), whole(raw, "topk_group"))
            else {
                bail!(
                    "{} routes by {method} without n_group and topk_group",
                    path.display()
                );
            };
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
                rank_by_top_two: method == "noaux_tc",
            })
        }
        other => bail!(
            "{} declares topk_method {other:?}; Ster implements greedy, group_limited_greedy and noaux_tc",
            path.display()
        ),
    };
    // K-EXAONE keeps the selection bias beside the router rather than in it.
    let bias = if model_type == "exaone_moe" {
        "mlp.e_score_correction_bias"
    } else {
        "mlp.gate.e_score_correction_bias"
    };
    routed.selection_bias = (method == "noaux_tc").then_some(bias);
    let scale = number(raw, "routed_scaling_factor");
    routed.routed_scale = if v3_router || !(normalize && routed.top_k > 1) {
        scale
    } else {
        None
    };
    Ok(routed)
}
