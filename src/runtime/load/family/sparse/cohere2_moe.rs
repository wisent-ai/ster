//! Cohere2-MoE.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{Architecture, ExpertLayout, NormKind, Scoring, SharedExpert, SharedForm};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, fits, listed_layers};
use crate::runtime::load::family::read::{flag, number, text, whole};

/// Cohere2-MoE (`cohere2_moe`): Cohere's parallel block (one
/// `input_layernorm` before attention and the feed-forward side by side)
/// with RMS norms, or LayerNorms without a bias when it states no
/// `rms_norm_eps`. Adjacent pairs rotate on the sliding-window layers
/// `layer_types` lists, whose window is `sliding_window + 1` keys as vLLM
/// counts it, and under `prefix_dense_sliding_window_pattern` 1 on the
/// dense prefix too; the other layers apply no rotation. The dense layers
/// (`mlp_layer_types`, else the first `first_k_dense_replace`) are
/// `prefix_dense_intermediate_size` wide; the rest route over
/// `num_experts` experts `intermediate_size` wide by `expert_selection_fn`
/// scores, renormalised under `norm_topk_prob`, beside
/// `num_shared_experts` shared ones whose sum with the routed ones is
/// halved under `shared_expert_combination_strategy` `average`.
pub(in crate::runtime::load::family) fn cohere2_moe(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    if flag(raw, "use_qk_norm") {
        bail!(
            "{} declares use_qk_norm; Ster implements Cohere2-MoE without query and key norms",
            path.display()
        );
    }
    if raw.get("use_gated_activation").and_then(Value::as_bool) == Some(false) {
        bail!(
            "{} declares use_gated_activation false; Ster implements Cohere2-MoE's gated feed-forward",
            path.display()
        );
    }
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!(
            "{} declares a Cohere2-MoE model without layer_types",
            path.display()
        );
    };
    let windowed = listed_layers(types, layers, path)?;
    architecture.norm = if flag(raw, "layer_norm") {
        NormKind::Layer { bias: false }
    } else {
        NormKind::Rms
    };
    architecture.parallel = true;
    architecture.interleaved_rotary = true;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    architecture.logits_multiplier = number(raw, "logit_scale");
    architecture.sliding_layers = windowed;
    architecture.sliding_window = whole(raw, "sliding_window").map(|window| window + 1);
    let dense = match raw.get("mlp_layer_types").and_then(Value::as_array) {
        Some(kinds) => {
            if kinds.len() != layers {
                bail!(
                    "{} lists {} mlp_layer_types for {layers} layers",
                    path.display(),
                    kinds.len()
                );
            }
            let mut set = 0u128;
            for (layer, kind) in kinds.iter().enumerate() {
                match kind.as_str() {
                    Some("dense") => set |= 1u128 << layer,
                    Some("sparse") => {}
                    other => bail!(
                        "{} declares layer {layer}'s feed-forward as {other:?}; Ster implements dense and sparse",
                        path.display()
                    ),
                }
            }
            set
        }
        None => {
            let first = whole(raw, "first_k_dense_replace").unwrap_or(0).min(layers);
            (0..first).fold(0u128, |set, layer| set | (1u128 << layer))
        }
    };
    let prefix = (0..layers)
        .take_while(|layer| dense & (1u128 << layer) != 0)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let forced = if whole(raw, "prefix_dense_sliding_window_pattern").unwrap_or(1) == 1 {
        prefix
    } else {
        0
    };
    architecture.unrotated_layers = every_layer(layers, path)? & !windowed & !forced;
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
    routed.scoring = match text(raw, "expert_selection_fn") {
        None | Some("softmax") => Scoring::Softmax,
        Some("sigmoid") => Scoring::Sigmoid,
        Some(other) => bail!(
            "{} declares expert_selection_fn {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    routed.shared = whole(raw, "num_shared_experts")
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    routed.average_shared = routed.shared.is_some()
        && match text(raw, "shared_expert_combination_strategy") {
            None | Some("sum") => false,
            Some("average") => true,
            Some(other) => bail!(
                "{} declares shared_expert_combination_strategy {other:?}; Ster implements sum and average",
                path.display()
            ),
        };
    architecture.experts = Some(routed);
    Ok(())
}
