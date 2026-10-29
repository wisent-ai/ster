//! AFMoE.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{
    Architecture, ExpertGroups, ExpertLayout, GateFunction, Names, QueryKeyNorm, Scoring,
    SharedExpert, SharedForm,
};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, listed_layers};
use crate::runtime::load::family::read::{flag, number, text, whole};

/// AFMoE (`afmoe`, Arcee Trinity): sandwich norms (`post_attention_layernorm`
/// over attention's output, `pre_mlp_layernorm` before the feed-forward,
/// `post_mlp_layernorm` over it), per-head query and key norms, attention's
/// output multiplied by the sigmoid of `self_attn.gate_proj`, and the
/// rotation on the sliding-window layers `layer_types` lists only. Under
/// `mup_enabled` the embeddings are multiplied by `sqrt(hidden_size)`. The
/// layers from `num_dense_layers` route over `num_experts` experts by
/// `score_func` scores that `mlp.expert_bias` moves to choose (within the
/// best `topk_group` of `n_group` groups), renormalised under `route_norm`
/// when the scores are sigmoids and scaled by `route_scale`, beside
/// `num_shared_experts` shared ones.
pub(in crate::runtime::load::family) fn afmoe(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let windowed = match raw.get("layer_types").and_then(Value::as_array) {
        Some(types) => listed_layers(types, layers, path)?,
        None => {
            let every = whole(raw, "global_attn_every_n_layers").unwrap_or(1).max(1);
            (0..layers)
                .filter(|layer| (layer + 1) % every != 0)
                .fold(0u128, |set, layer| set | (1u128 << layer))
        }
    };
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = windowed;
    architecture.unrotated_layers = every_layer(layers, path)? & !windowed;
    architecture.output_norms = true;
    architecture.names = Names::PANGU_SANDWICH;
    architecture.query_key_norm = QueryKeyNorm::PerHead;
    architecture.attention_gate = Some(GateFunction::Sigmoid);
    if flag(raw, "mup_enabled") {
        architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
    }
    let first = whole(raw, "num_dense_layers").unwrap_or(0).min(layers);
    let dense = (0..first).fold(0u128, |set, layer| set | (1u128 << layer));
    let sigmoid = match text(raw, "score_func") {
        None | Some("sigmoid") => true,
        Some("softmax") => false,
        Some(other) => bail!(
            "{} declares score_func {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    let normalize = sigmoid
        && raw
            .get("route_norm")
            .and_then(Value::as_bool)
            .unwrap_or(true);
    let mut routed = experts(
        raw,
        "num_experts",
        "moe_intermediate_size",
        normalize,
        ExpertLayout::HyV3,
        dense,
        path,
    )?;
    routed.scoring = if sigmoid {
        Scoring::Sigmoid
    } else {
        Scoring::Softmax
    };
    routed.routed_scale = number(raw, "route_scale");
    routed.selection_bias = Some("mlp.expert_bias");
    routed.groups = match (whole(raw, "n_group"), whole(raw, "topk_group")) {
        (Some(groups), Some(chosen_groups)) if groups > 1 => {
            if routed.count % groups != 0 || chosen_groups > groups {
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
    routed.shared = whole(raw, "num_shared_experts")
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
