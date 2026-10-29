//! Mixtures of experts: the counts, widths and dense layers a config states.

pub(in crate::runtime::load::family) mod deepseek;
pub(in crate::runtime::load::family) mod index;
pub(in crate::runtime::load::family) mod step;

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{ExpertLayout, MixtureOfExperts, Scoring};
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{uniform, whole};

/// The mixture of experts a config declares: how many under `count_key`, how
/// many per token under `num_experts_per_tok`, each one's width under
/// `width_key`, each stated once or once per layer.
pub(in crate::runtime::load::family) fn experts(
    raw: &Value,
    count_key: &str,
    width_key: &str,
    normalize: bool,
    layout: ExpertLayout,
    dense_layers: u128,
    path: &Path,
) -> Result<MixtureOfExperts> {
    let (Some(count), Some(top_k), Some(intermediate)) = (
        uniform(raw, count_key, path)?,
        uniform(raw, "num_experts_per_tok", path)?,
        uniform(raw, width_key, path)?,
    ) else {
        bail!(
            "{} declares a mixture of experts without {count_key}, num_experts_per_tok and {width_key}",
            path.display()
        );
    };
    if top_k == 0 || top_k > count {
        bail!(
            "{} routes each token to {top_k} of {count} experts; it must be at least one and at most all of them",
            path.display()
        );
    }
    Ok(MixtureOfExperts {
        count,
        top_k,
        intermediate,
        normalize,
        shared: None,
        layout,
        dense_layers,
        scoring: Scoring::Softmax,
        groups: None,
        selection_bias: None,
        routed_scale: None,
        swiglu_limit: None,
        identity_experts: 0,
        average_shared: false,
        weight_input: false,
        latent: None,
        hash_layers: 0,
        routed_activation: None,
    })
}

/// Qwen MoE's dense layers: those `mlp_layer_types` calls `dense` when the
/// config lists them (Mellum), otherwise those in `mlp_only_layers` and
/// those whose position is not a multiple of `decoder_sparse_step`.
pub(in crate::runtime::load::family) fn qwen_dense_layers(
    raw: &Value,
    layers: usize,
    path: &Path,
) -> Result<u128> {
    fits(layers, path)?;
    if let Some(kinds) = raw.get("mlp_layer_types").and_then(Value::as_array) {
        if kinds.len() != layers {
            bail!(
                "{} lists {} mlp_layer_types for {layers} layers",
                path.display(),
                kinds.len()
            );
        }
        let mut dense = 0u128;
        for (layer, kind) in kinds.iter().enumerate() {
            match kind.as_str() {
                Some("dense") => dense |= 1u128 << layer,
                Some("sparse") => {}
                other => bail!(
                    "{} names layer {layer}'s feed-forward {other:?}; mlp_layer_types entries are dense or sparse",
                    path.display()
                ),
            }
        }
        return Ok(dense);
    }
    let step = whole(raw, "decoder_sparse_step").unwrap_or(1).max(1);
    let listed: Vec<usize> = raw
        .get("mlp_only_layers")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|v| v.as_u64().map(|v| v as usize))
                .collect()
        })
        .unwrap_or_default();
    Ok((0..layers)
        .filter(|layer| listed.contains(layer) || (layer + 1) % step != 0)
        .fold(0, |set, layer| set | (1u128 << layer)))
}
