//! K2-Horizon.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, GateFunction, MixtureOfExperts, Scoring, SharedExpert, SharedForm,
};
use crate::runtime::load::family::experts::{experts, qwen_dense_layers};
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{flag, number, text, whole};

/// A per-layer list of numbers under `key`, one value for the
/// full-attention layers and one for the sliding-window layers in
/// `windowed`, as `(full, sliding)`; refused when either kind's values
/// differ.
pub(in crate::runtime::load::family) fn per_kind(
    raw: &Value,
    key: &str,
    windowed: u128,
    layers: usize,
    path: &Path,
) -> Result<(Option<f64>, Option<f64>)> {
    let Some(list) = raw.get(key).filter(|list| !list.is_null()) else {
        return Ok((None, None));
    };
    let values: Vec<f64> = list
        .as_array()
        .and_then(|entries| entries.iter().map(Value::as_f64).collect())
        .with_context(|| {
            format!(
                "{} declares {key} that is not a list of numbers",
                path.display()
            )
        })?;
    if values.len() != layers {
        bail!(
            "{} lists {} {key} values for {layers} layers",
            path.display(),
            values.len()
        );
    }
    fits(layers, path)?;
    let mut kinds = [None, None];
    for (layer, value) in values.into_iter().enumerate() {
        let slot = usize::from(windowed & (1u128 << layer) != 0);
        match kinds[slot] {
            Some(seen) if seen != value => bail!(
                "{} lists more than one {key} value for its {} layers; Ster gives each attention kind one",
                path.display(),
                if slot == 1 {
                    "sliding-window"
                } else {
                    "full-attention"
                }
            ),
            _ => kinds[slot] = Some(value),
        }
    }
    Ok((kinds[0], kinds[1]))
}

/// K2-Horizon (`k2_horizon`): Llama's block with hidden-width RMS norms
/// over `layernorm_num_groups` equal groups, an elementwise attention gate
/// (`self_attn.gate_proj` through `attention_gate_func`, `silu` or
/// `softplus` with β = ln 2) and biased projections under `attention_bias`.
/// The layers neither in `mlp_only_layers` nor off `decoder_sparse_step`
/// route over `num_experts` experts by `router_score_func` scores that
/// `mlp.gate.bias` moves to choose (`moe_gate_bias`), renormalised under
/// `norm_topk_prob` and scaled by `router_scaling_factor`, beside
/// `num_shared_experts` shared ones; under `mova_num_experts` those layers
/// take each value from `mova_num_experts_per_tok` value experts routed the
/// same way. A rotation narrower than the head (`rope_head_dim`, whose
/// channels the checkpoint interleaves) and per-head query and key norms
/// (`query_key_norm`) are refused.
pub(in crate::runtime::load::family) fn k2_horizon(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let head_dim = architecture.head_dim;
    if let Some(rotated) = whole(raw, "rope_head_dim").filter(|rotated| *rotated != head_dim) {
        bail!(
            "{} rotates {rotated} of {head_dim} channels per head (rope_head_dim), interleaved through the head; Ster rotates whole K2-Horizon heads only",
            path.display()
        );
    }
    if flag(raw, "query_key_norm") {
        bail!(
            "{} declares query_key_norm, one scale per channel of every head; Ster implements K2-Horizon without query and key norms",
            path.display()
        );
    }
    let groups = whole(raw, "layernorm_num_groups").unwrap_or(1).max(1);
    if llama.hidden_size % groups != 0 {
        bail!(
            "{} splits its {} hidden channels into {groups} norm groups unevenly",
            path.display(),
            llama.hidden_size
        );
    }
    architecture.norm_groups = groups;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    architecture.attention_gate = match text(raw, "attention_gate_func") {
        None => None,
        Some("silu") => Some(GateFunction::Silu),
        Some("softplus") => Some(GateFunction::Softplus),
        Some(other) => bail!(
            "{} declares attention_gate_func {other:?}; Ster implements silu and softplus",
            path.display()
        ),
    };
    if whole(raw, "num_experts").unwrap_or(0) == 0 {
        return Ok(());
    }
    let scoring = match text(raw, "router_score_func") {
        None | Some("softmax") => Scoring::Softmax,
        Some("sigmoid") => Scoring::Sigmoid,
        Some(other) => bail!(
            "{} declares router_score_func {other:?}; Ster implements softmax and sigmoid expert scores",
            path.display()
        ),
    };
    let biased = flag(raw, "moe_gate_bias");
    let mut routed = experts(
        raw,
        "num_experts",
        "moe_intermediate_size",
        flag(raw, "norm_topk_prob"),
        ExpertLayout::Qwen,
        qwen_dense_layers(raw, layers, path)?,
        path,
    )?;
    routed.scoring = scoring;
    routed.routed_scale = number(raw, "router_scaling_factor");
    routed.selection_bias = biased.then_some("mlp.gate.bias");
    routed.shared = whole(raw, "num_shared_experts")
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    if let Some(count) = whole(raw, "mova_num_experts").filter(|count| *count > 0) {
        let top_k = whole(raw, "mova_num_experts_per_tok").unwrap_or(0);
        if top_k == 0 || top_k > count {
            bail!(
                "{} routes each value to {top_k} of {count} value experts (mova_num_experts_per_tok); it must be at least one and at most all of them",
                path.display()
            );
        }
        architecture.value_experts = Some(MixtureOfExperts {
            count,
            top_k,
            intermediate: llama
                .num_key_value_heads
                .unwrap_or(llama.num_attention_heads)
                * head_dim,
            normalize: top_k > 1,
            shared: None,
            layout: ExpertLayout::Mova,
            dense_layers: routed.dense_layers,
            scoring,
            groups: None,
            selection_bias: biased.then_some("self_attn.v_router.bias"),
            routed_scale: routed.routed_scale,
            swiglu_limit: None,
            identity_experts: 0,
            average_shared: false,
            weight_input: false,
            latent: None,
            hash_layers: 0,
            routed_activation: None,
        });
    }
    architecture.experts = Some(routed);
    Ok(())
}
