//! Gemma 3n.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Activation, Architecture, Names, PerLayerInputSpec, QueryKeyNorm};
use crate::runtime::load::family::read::layers::listed_layers;
use crate::runtime::load::family::read::{number, whole};

/// Gemma 3n's text decoder (`gemma3n_text`, inside `gemma3n` below
/// `model.language_model`): Gemma 4's attention — per-head query and key
/// norms scaling by their weight itself, a scale-free norm over each value,
/// scores left undivided, sliding-window layers as `layer_types` lists them
/// rotating at `rope_local_base_freq`, the last `num_kv_shared_layers`
/// layers reusing an earlier layer's keys and values — and its per-layer
/// inputs, with three additions: `altup_num_inputs` AltUp streams, Laurel
/// (`laurel_rank`) beside attention, and on the layers whose
/// `activation_sparsity_pattern` entry is above zero a feed-forward gate cut
/// at its mean plus the normal quantile of that entry in standard
/// deviations.
pub(in crate::runtime::load::family) fn gemma3n(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!(
            "{} declares a Gemma 3n model without layer_types",
            path.display()
        );
    };
    if whole(raw, "altup_active_idx").unwrap_or(0) != 0 {
        bail!(
            "{} runs its blocks on AltUp stream {:?}; Ster runs them on stream 0",
            path.display(),
            raw.get("altup_active_idx")
        );
    }
    if raw.get("altup_correct_scale").and_then(Value::as_bool) == Some(false) {
        bail!(
            "{} declares altup_correct_scale false; Ster scales the corrected stream as Gemma 3n does",
            path.display()
        );
    }
    let streams = whole(raw, "altup_num_inputs")
        .filter(|streams| *streams >= 1)
        .with_context(|| {
            format!(
                "{} declares a Gemma 3n model without altup_num_inputs",
                path.display()
            )
        })?;
    architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
    architecture.activation = Activation::GeluTanh;
    architecture.output_norms = true;
    architecture.names = Names::GEMMA2;
    architecture.query_key_norm = QueryKeyNorm::PerHead;
    architecture.value_norm = true;
    architecture.score_divisor = 1.0;
    architecture.final_softcap = number(raw, "final_logit_softcapping");
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = listed_layers(types, layers, path)?;
    architecture.local_rope_theta = number(raw, "rope_local_base_freq").map(|theta| theta as f32);
    let shared = whole(raw, "num_kv_shared_layers").unwrap_or(0);
    if shared > 0 {
        if shared >= layers {
            bail!(
                "{} shares keys and values on {shared} of {layers} layers, leaving none to produce them",
                path.display()
            );
        }
        architecture.shared_key_values = Some(layers - shared);
    }
    if let Some(width) = whole(raw, "hidden_size_per_layer_input").filter(|width| *width > 0) {
        architecture.per_layer_input = Some(PerLayerInputSpec {
            width,
            vocab: whole(raw, "vocab_size_per_layer_input").unwrap_or(llama.vocab_size),
        });
    }
    architecture.altup_streams = Some(streams);
    architecture.laurel_rank = whole(raw, "laurel_rank").filter(|rank| *rank > 0);
    if let Some(pattern) = raw
        .get("activation_sparsity_pattern")
        .and_then(Value::as_array)
    {
        let shares: Vec<f64> = pattern
            .iter()
            .map(Value::as_f64)
            .collect::<Option<_>>()
            .with_context(|| {
                format!(
                    "{} declares activation_sparsity_pattern that is not a list of numbers",
                    path.display()
                )
            })?;
        if shares.len() != layers {
            bail!(
                "{} lists {} activation_sparsity_pattern entries for {layers} layers",
                path.display(),
                shares.len()
            );
        }
        let sparse: Vec<(usize, f64)> = shares
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, share)| *share > 0.0)
            .collect();
        if let Some(&(_, share)) = sparse.first() {
            if sparse.iter().any(|(_, other)| *other != share) || share >= 1.0 {
                bail!(
                    "{} lists sparse layers with different or whole sparsity in activation_sparsity_pattern; Ster cuts every sparse layer at one quantile below one",
                    path.display()
                );
            }
            // The standard normal quantile of `share`: √2 · erf⁻¹(2·share − 1).
            let multiplier =
                std::f64::consts::SQRT_2 * candle_core::cpu::erf::erf_inv(2.0 * share - 1.0);
            let layers = sparse
                .iter()
                .fold(0u128, |set, (layer, _)| set | (1u128 << layer));
            architecture.activation_sparsity = Some((multiplier, layers));
        }
    }
    Ok(())
}
