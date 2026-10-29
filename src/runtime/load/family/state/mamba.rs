//! Mamba, Falcon-Mamba, Jamba and Mamba-2.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, Names, ParameterNorm, Positions, StateSpaceSpec, StructuredSpec,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, number, whole};

/// What the families this file reads add to the Llama block; [`super::super::family`]
/// sends only their `model_type` values here.
pub(in crate::runtime::load::family) fn configure(
    at: &At,
    architecture: &mut Architecture,
) -> Result<()> {
    let At {
        model_type,
        raw,
        llama,
        layers,
        path,
        ..
    } = *at;
    match model_type {
        "mamba" | "falcon_mamba" => {
            // Mamba: every block is a norm and a selective state-space
            // mixer; Falcon-Mamba adds a scale-free RMS norm on the step and
            // the input and output matrices.
            let hidden = llama.hidden_size;
            let inner = whole(raw, "intermediate_size")
                .or_else(|| whole(raw, "expand").map(|expand| expand * hidden));
            let (Some(inner), Some(state), Some(kernel)) =
                (inner, whole(raw, "state_size"), whole(raw, "conv_kernel"))
            else {
                bail!(
                    "{} declares a state-space model without intermediate_size (or expand), state_size and conv_kernel",
                    path.display()
                );
            };
            let step_rank = match raw.get("time_step_rank") {
                Some(Value::String(rule)) if rule == "auto" => {
                    hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK)
                }
                _ => whole(raw, "time_step_rank")
                    .unwrap_or(hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK)),
            };
            architecture.names = Names::MAMBA;
            architecture.positions = Positions::None;
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank,
                projection_bias: flag(raw, "use_bias"),
                convolution_bias: raw
                    .get("use_conv_bias")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                parameter_norm: if model_type == "falcon_mamba" {
                    ParameterNorm::Bare(number(raw, "mixer_rms_eps").unwrap_or(llama.rms_norm_eps))
                } else {
                    ParameterNorm::None
                },
                layers: every_layer(layers, path)?,
                feed_forward: false,
                structured: None,
            });
        }
        "jamba" => {
            // Jamba: attention on every `attn_layer_period`-th layer from
            // `attn_layer_offset` — with no position signal at all — and
            // Mamba elsewhere with weighted norms on its step and matrices;
            // every layer has a feed-forward, routed over `num_experts` on
            // every `expert_layer_period`-th layer from `expert_layer_offset`.
            fits(layers, path)?;
            let periodic = |period: &str, offset: &str| -> u128 {
                let period = whole(raw, period).unwrap_or(1).max(1);
                let offset = whole(raw, offset).unwrap_or(0);
                (0..layers)
                    .filter(|layer| layer % period == offset)
                    .fold(0, |set, layer| set | (1u128 << layer))
            };
            let attention = periodic("attn_layer_period", "attn_layer_offset");
            let routed = periodic("expert_layer_period", "expert_layer_offset");
            let hidden = llama.hidden_size;
            let inner = whole(raw, "mamba_expand").map(|expand| expand * hidden);
            let (Some(inner), Some(state), Some(kernel)) = (
                inner,
                whole(raw, "mamba_d_state"),
                whole(raw, "mamba_d_conv"),
            ) else {
                bail!(
                    "{} declares a Jamba model without mamba_expand, mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            let step_rank = match raw.get("mamba_dt_rank") {
                Some(Value::String(rule)) if rule == "auto" => {
                    hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK)
                }
                _ => whole(raw, "mamba_dt_rank")
                    .unwrap_or(hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK)),
            };
            architecture.names = Names::JAMBA;
            architecture.positions = Positions::None;
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank,
                projection_bias: flag(raw, "mamba_proj_bias"),
                convolution_bias: raw
                    .get("mamba_conv_bias")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                parameter_norm: ParameterNorm::Weighted(llama.rms_norm_eps),
                layers: every_layer(layers, path)? & !attention,
                feed_forward: true,
                structured: None,
            });
            if whole(raw, "num_experts").is_some_and(|count| count > 1) {
                architecture.experts = Some(experts(
                    raw,
                    "num_experts",
                    "intermediate_size",
                    false,
                    ExpertLayout::Jamba,
                    every_layer(layers, path)? & !routed,
                    path,
                )?);
            }
        }
        "mamba2" => {
            // Mamba-2: every block is a norm and the structured mixer.
            let heads = structured(raw, "num_heads", "head_dim", "n_groups", path)?;
            let inner = heads.heads * heads.head_dim;
            let (Some(state), Some(kernel)) = (whole(raw, "state_size"), whole(raw, "conv_kernel"))
            else {
                bail!(
                    "{} declares a Mamba-2 model without state_size and conv_kernel",
                    path.display()
                );
            };
            architecture.names = Names::MAMBA;
            architecture.positions = Positions::None;
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "use_bias"),
                convolution_bias: raw
                    .get("use_conv_bias")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                layers: every_layer(layers, path)?,
                feed_forward: false,
                structured: Some(heads),
            });
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// Mamba's `time_step_rank: "auto"` is the model width over this, rounded up,
/// as Transformers' `MambaConfig` computes it.
pub(in crate::runtime::load::family) const MAMBA_WIDTH_PER_STEP_RANK: usize = 16;

/// Mamba-2's head layout from the keys a family spells it with: the head
/// count, the head width and the group count (one when unstated), and
/// `time_step_limit` (unbounded when unstated or stated as non-finite).
pub(in crate::runtime::load::family) fn structured(
    raw: &Value,
    heads_key: &str,
    width_key: &str,
    groups_key: &str,
    path: &Path,
) -> Result<StructuredSpec> {
    let (Some(heads), Some(head_dim)) = (whole(raw, heads_key), whole(raw, width_key)) else {
        bail!(
            "{} declares a Mamba-2 mixer without {heads_key} and {width_key}",
            path.display()
        );
    };
    let groups = whole(raw, groups_key).unwrap_or(1);
    if groups == 0 || heads % groups != 0 {
        bail!(
            "{} splits {heads} Mamba-2 heads into {groups} groups; the groups must divide the heads evenly",
            path.display()
        );
    }
    let limit = raw.get("time_step_limit").and_then(Value::as_array);
    let bound = |index: usize, unstated: f64| {
        limit
            .and_then(|limit| limit.get(index))
            .and_then(Value::as_f64)
            .unwrap_or(unstated)
    };
    Ok(StructuredSpec {
        heads,
        head_dim,
        groups,
        step_limit: (bound(0, 0.0), bound(1, f64::INFINITY)),
        input_scale: 1.0,
        projection_scales: None,
        gated_norm: true,
    })
}
