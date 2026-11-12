//! Mixtures of experts: the counts, widths and dense layers a config states.

pub(in crate::runtime::load::family) mod deepseek;
pub(in crate::runtime::load::family) mod index;
pub(in crate::runtime::load::family) mod step;

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::model::{ExpertLayout, MixtureOfExperts, Scoring};
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{list, uniform, whole_or};

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
        refused!(
            Incomplete,
            in raw,
            [count_key, "num_experts_per_tok", width_key],
            "{} declares a mixture of experts without {count_key}, num_experts_per_tok and {width_key}",
            path.display()
        );
    };
    if top_k == 0 || top_k > count {
        refused!(
            OutOfRange,
            in raw,
            [count_key, "num_experts_per_tok"],
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
    if let Some(kinds) = list(raw, "mlp_layer_types", path)? {
        if kinds.len() != layers {
            refused!(
                Contradictory,
                in raw,
                ["mlp_layer_types", "num_hidden_layers"],
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
                other => refused!(
                    Malformed,
                    in raw,
                    ["mlp_layer_types"],
                    "{} names layer {layer}'s feed-forward {other:?}; mlp_layer_types entries are dense or sparse",
                    path.display()
                ),
            }
        }
        return Ok(dense);
    }
    let step = whole_or(raw, "decoder_sparse_step", 1, path)?.max(1);
    let mut listed: Vec<usize> = Vec::new();
    for entry in list(raw, "mlp_only_layers", path)?.into_iter().flatten() {
        match entry.as_u64() {
            Some(layer) => listed.push(layer as usize),
            None => refused!(
                Malformed,
                in raw,
                ["mlp_only_layers"],
                "{} lists mlp_only_layers entry {entry}, which is not a layer number",
                path.display()
            ),
        }
    }
    Ok((0..layers)
        .filter(|layer| listed.contains(layer) || (layer + 1) % step != 0)
        .fold(0, |set, layer| set | (1u128 << layer)))
}
