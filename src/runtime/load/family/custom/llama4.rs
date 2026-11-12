//! Llama 4 and its query temperature.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, Names, QueryKeyNorm, QueryTemperature, Scoring, SharedExpert,
    SharedForm,
};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::every_layer;
use crate::runtime::load::family::read::{
    describe, flag, flag_or, list, mistyped, number, number_or, stated, whole, whole_or, within,
};

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
    stated::<&serde_json::Map<String, Value>>(raw, "llama_4_scaling", path)?;
    let read = |key: &str| -> Result<Option<f64>> {
        if let Some(value) = number(raw, &format!("llama_4_scaling.{key}"), path)? {
            return Ok(Some(value));
        }
        match scaling {
            Some(scaling) => within(scaling, "rope_scaling", key, path),
            None => Ok(None),
        }
    };
    let beta = match read("llama_4_scaling_beta")? {
        Some(beta) => Some(beta),
        None => read("beta")?,
    };
    let original = read("original_max_position_embeddings")?.map(|original| original as usize);
    Ok(match (beta, original) {
        (Some(beta), Some(interval)) if interval > 0 => Some(QueryTemperature {
            beta,
            interval,
            shift: 0,
            layers: every_layer(layers, path)?,
        }),
        (Some(_), _) => refused!(
            Incomplete,
            [
                "llama_4_scaling" => describe(raw.get("llama_4_scaling")),
                "rope_scaling" => describe(scaling),
            ],
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
    architecture.query_key_value_bias = flag(raw, "attention_bias", path)?;
    architecture.output_bias = architecture.query_key_value_bias;
    let rotating = match list(raw, "no_rope_layers", path)?.filter(|flags| !flags.is_empty()) {
        Some(flags) => {
            if flags.len() != layers {
                refused!(
                    Contradictory,
                    ["no_rope_layers" => format!("{} entries", flags.len()), "num_hidden_layers" => layers],
                    "{} lists {} no_rope_layers entries for {layers} layers",
                    path.display(),
                    flags.len()
                );
            }
            let mut rotating = 0u128;
            for (layer, flag) in flags.iter().enumerate() {
                match (flag.as_u64(), flag.as_bool()) {
                    (Some(1), _) | (_, Some(true)) => rotating |= 1u128 << layer,
                    (Some(0), _) | (_, Some(false)) => {}
                    _ => {
                        let key = format!("no_rope_layers[{layer}]");
                        return Err(mistyped(&key, flag, "1, 0, true or false", path));
                    }
                }
            }
            rotating
        }
        None => (0..layers)
            .filter(|layer| (layer + 1) % LLAMA4_NOPE_INTERVAL != 0)
            .fold(0u128, |set, layer| set | (1u128 << layer)),
    };
    architecture.unrotated_layers = every & !rotating;
    if let Some(chunk) = whole(raw, "attention_chunk_size", path)?.filter(|chunk| *chunk > 0) {
        architecture.sliding_window = Some(chunk);
        architecture.sliding_layers = rotating;
        architecture.chunk_lookback = Some(0);
    }
    if flag(raw, "use_qk_norm", path)? {
        architecture.query_key_norm = QueryKeyNorm::Unscaled;
        architecture.norm_after_rotary = true;
    }
    if flag_or(raw, "attn_temperature_tuning", true, path)? {
        architecture.query_temperature = Some(QueryTemperature {
            beta: number_or(raw, "attn_scale", LLAMA4_ATTENTION_SCALE, path)?,
            interval: whole_or(raw, "floor_scale", LLAMA4_FLOOR_SCALE, path)?
                .max(1),
            shift: 1,
            layers: every & !rotating,
        });
    }
    let routed_layers = match list(raw, "moe_layers", path)? {
        Some(listed) => {
            let mut set = 0u128;
            for entry in listed {
                match entry.as_u64().map(|layer| layer as usize) {
                    Some(layer) if layer < layers => set |= 1u128 << layer,
                    _ => refused!(
                        OutOfRange,
                        in raw,
                        ["moe_layers"],
                        "{} lists moe_layers entry {entry}, which names no layer below {layers}",
                        path.display()
                    ),
                }
            }
            set
        }
        None => {
            let step = whole_or(raw, "interleave_moe_layer_step", 1, path)?.max(1);
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
