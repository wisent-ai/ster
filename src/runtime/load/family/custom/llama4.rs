//! Llama 4 and its query temperature.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, Names, QueryKeyNorm, QueryTemperature, Scoring, SharedExpert,
    SharedForm,
};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::every_layer;
use crate::runtime::load::family::read::{flag, number, whole};

/// How often Llama 4 skips the rotation when its config lists no
/// `no_rope_layers`: every fourth layer, counting from one (Transformers'
/// `Llama4TextConfig`, `no_rope_layer_interval=4`).
pub(in crate::runtime::load::family) const LLAMA4_NOPE_INTERVAL: usize = 4;

/// Llama 4's attention temperature when its config leaves it out:
/// `attn_scale` and `floor_scale` (Transformers' `Llama4TextConfig`
/// defaults, 0.1 and 8192).
pub(in crate::runtime::load::family) const LLAMA4_ATTENTION_SCALE: f64 = 0.1;

pub(in crate::runtime::load::family) const LLAMA4_FLOOR_SCALE: usize = 8192;

/// Llama 4's query temperature as Ministral 3 and Mistral Large 3 state it
/// (vLLM's `llama_4_scaling`): every query past each
/// `original_max_position_embeddings` positions multiplied by
/// `1 + beta · ln(1 + floor(position / original))`, the two read from the
/// `llama_4_scaling` block or, as Transformers writes them, beside the
/// rotation (`llama_4_scaling_beta`). None when no beta is stated.
pub(in crate::runtime::load::family) fn llama4_temperature(
    raw: &Value,
    scaling: Option<&Value>,
    layers: usize,
    path: &Path,
) -> Result<Option<QueryTemperature>> {
    let stated = |key: &str| {
        raw.get("llama_4_scaling")
            .and_then(|block| block.get(key))
            .or_else(|| scaling.and_then(|scaling| scaling.get(key)))
            .and_then(Value::as_f64)
    };
    let beta = stated("llama_4_scaling_beta").or_else(|| stated("beta"));
    let original = stated("original_max_position_embeddings").map(|original| original as usize);
    Ok(match (beta, original) {
        (Some(beta), Some(interval)) if interval > 0 => Some(QueryTemperature {
            beta,
            interval,
            shift: 0,
            layers: every_layer(layers, path)?,
        }),
        (Some(_), _) => bail!(
            "{} states llama_4_scaling_beta without original_max_position_embeddings to count positions by",
            path.display()
        ),
        (None, _) => None,
    })
}

/// Llama 4 (`llama4_text`, also inside `llama4`): Llama's attention, the
/// layers `no_rope_layers` marks (every fourth by default) unrotated and
/// attending fully, the rotating ones attending by chunks of
/// `attention_chunk_size` with their queries and keys normed without a
/// weight after the rotation under `use_qk_norm`; under
/// `attn_temperature_tuning` the unrotated layers' queries at position `p`
/// are multiplied by `1 + attn_scale · ln(1 + floor((p + 1) / floor_scale))`.
/// The layers `moe_layers` lists (else every `interleave_moe_layer_step`-th)
/// route each token to `num_experts_per_tok` of `num_local_experts` experts
/// by sigmoid scores that weigh the expert's input, beside a shared expert;
/// the others are dense, `intermediate_size_mlp` wide.
pub(in crate::runtime::load::family) fn llama4(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let every = every_layer(layers, path)?;
    architecture.names = Names::LLAMA4;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    let rotating = match raw
        .get("no_rope_layers")
        .and_then(Value::as_array)
        .filter(|flags| !flags.is_empty())
    {
        Some(flags) => {
            if flags.len() != layers {
                bail!(
                    "{} lists {} no_rope_layers entries for {layers} layers",
                    path.display(),
                    flags.len()
                );
            }
            flags
                .iter()
                .enumerate()
                .filter(|(_, flag)| flag.as_u64() == Some(1) || flag.as_bool() == Some(true))
                .fold(0u128, |set, (layer, _)| set | (1u128 << layer))
        }
        None => (0..layers)
            .filter(|layer| (layer + 1) % LLAMA4_NOPE_INTERVAL != 0)
            .fold(0u128, |set, layer| set | (1u128 << layer)),
    };
    architecture.unrotated_layers = every & !rotating;
    if let Some(chunk) = whole(raw, "attention_chunk_size").filter(|chunk| *chunk > 0) {
        architecture.sliding_window = Some(chunk);
        architecture.sliding_layers = rotating;
        architecture.chunk_lookback = Some(0);
    }
    if flag(raw, "use_qk_norm") {
        architecture.query_key_norm = QueryKeyNorm::Unscaled;
        architecture.norm_after_rotary = true;
    }
    if raw
        .get("attn_temperature_tuning")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        architecture.query_temperature = Some(QueryTemperature {
            beta: number(raw, "attn_scale").unwrap_or(LLAMA4_ATTENTION_SCALE),
            interval: whole(raw, "floor_scale")
                .unwrap_or(LLAMA4_FLOOR_SCALE)
                .max(1),
            shift: 1,
            layers: every & !rotating,
        });
    }
    let routed_layers = match raw.get("moe_layers").and_then(Value::as_array) {
        Some(listed) => {
            let mut set = 0u128;
            for entry in listed {
                match entry.as_u64().map(|layer| layer as usize) {
                    Some(layer) if layer < layers => set |= 1u128 << layer,
                    _ => bail!(
                        "{} lists moe_layers entry {entry}, which names no layer below {layers}",
                        path.display()
                    ),
                }
            }
            set
        }
        None => {
            let step = whole(raw, "interleave_moe_layer_step").unwrap_or(1).max(1);
            (step - 1..layers)
                .step_by(step)
                .fold(0u128, |set, layer| set | (1u128 << layer))
        }
    };
    let mut routed = experts(
        raw,
        "num_local_experts",
        "moe_intermediate_size",
        false,
        ExpertLayout::Llama4,
        every & !routed_layers,
        path,
    )?;
    routed.scoring = Scoring::Sigmoid;
    routed.weight_input = true;
    routed.shared = Some(SharedExpert {
        intermediate: routed.intermediate,
        module: "feed_forward.shared_expert",
        gated: false,
        form: SharedForm::GateUpDown,
    });
    architecture.experts = Some(routed);
    Ok(())
}
