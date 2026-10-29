//! Sarvam-MLA.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{Architecture, ExpertGroups, ExpertLayout, Scoring, SharedExpert, SharedForm};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{number, text, whole};

/// The routed scale Sarvam's latent model uses when its config leaves
/// `routed_scaling_factor` out (vLLM's `sarvam.py`,
/// `getattr(config, "routed_scaling_factor", 2.5)`).
pub(in crate::runtime::load::family) const SARVAM_ROUTED_SCALE: f64 = 2.5;

/// Sarvam-105B (`sarvam_mla`): DeepSeek's latent attention, rotating
/// adjacent pairs, and a router over `num_experts` experts on the layers
/// from `first_k_dense_replace` (default one) at every `moe_layer_freq`-th:
/// `score_function` scores (sigmoid by default) that
/// `mlp.gate.e_score_correction_bias` moves under
/// `moe_router_enable_expert_bias` (on by default), within the best
/// `topk_group` of `n_group` groups when both are stated, renormalised
/// unless `norm_topk_prob` is false and scaled by `routed_scaling_factor`,
/// beside `num_shared_experts` shared experts
/// (`moe_shared_expert_intermediate_size`, else `moe_intermediate_size`,
/// wide each).
pub(in crate::runtime::load::family) fn sarvam_mla(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    if architecture.latent.is_none() {
        bail!(
            "{} declares a Sarvam MLA model without kv_lora_rank",
            path.display()
        );
    }
    fits(layers, path)?;
    architecture.interleaved_rotary = true;
    let first = whole(raw, "first_k_dense_replace").unwrap_or(1);
    let frequency = whole(raw, "moe_layer_freq").unwrap_or(1).max(1);
    let dense = (0..layers)
        .filter(|layer| *layer < first || (layer - first) % frequency != 0)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let normalize = raw
        .get("norm_topk_prob")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mut routed = experts(
        raw,
        "num_experts",
        "moe_intermediate_size",
        normalize,
        ExpertLayout::Qwen,
        dense,
        path,
    )?;
    routed.scoring = match text(raw, "score_function") {
        None | Some("sigmoid") => Scoring::Sigmoid,
        Some("softmax") => Scoring::Softmax,
        Some(other) => bail!(
            "{} declares score_function {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    let biased = raw
        .get("moe_router_enable_expert_bias")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    routed.selection_bias = biased.then_some("mlp.gate.e_score_correction_bias");
    routed.routed_scale = Some(number(raw, "routed_scaling_factor").unwrap_or(SARVAM_ROUTED_SCALE));
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
                rank_by_top_two: biased,
            })
        }
        _ => None,
    };
    let shared_width =
        whole(raw, "moe_shared_expert_intermediate_size").unwrap_or(routed.intermediate);
    routed.shared = Some(whole(raw, "num_shared_experts").unwrap_or(1))
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * shared_width,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    architecture.experts = Some(routed);
    Ok(())
}
