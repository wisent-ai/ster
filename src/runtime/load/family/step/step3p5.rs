//! Step-3.5.

use std::path::Path;

use anyhow::Result;
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, GateFunction, GlobalAttention, QueryKeyNorm, Scoring};
use crate::runtime::load::family::experts::step::{step_experts, step_limits};
use crate::runtime::load::family::read::layers::listed_layers;
use crate::runtime::load::family::read::{flag, list, mistyped, number, stated, text, whole};
use crate::runtime::load::family::step::k2::per_kind;

/// Step 3.5 (`step3p5`): RMS norms scaled by `1 + weight` throughout (vLLM's
/// `GemmaRMSNorm`), per-head query and key norms, and a sigmoid head-wise
/// gate (`g_proj`) under `use_head_wise_attn_gate`. Its layers alternate
/// full and `sliding_window` attention as `layer_types` lists; the
/// sliding-window layers' head counts come from `attention_other_setting`,
/// each kind rotates its own `partial_rotary_factors` share by halves at
/// its own base, `rope_scaling` reaches the full-attention layers only
/// (`yarn_only_types`), and `use_rope_layers` can leave layers unrotated.
/// Step's experts route by `moe_router_activation` scores that
/// `moe.router_bias` moves to choose and `moe_router_scaling_factor` scales,
/// each routed and shared expert's SwiGLU clamped on the layers
/// `swiglu_limits` and `swiglu_limits_shared` give a limit.
pub(in crate::runtime::load::family) fn step3p5(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(types) = list(raw, "layer_types", path)? else {
        refused!(
            Incomplete,
            in raw,
            ["layer_types"],
            "{} declares a Step 3.5 model without layer_types",
            path.display()
        );
    };
    let windowed = listed_layers(types, layers, path)?;
    if stated::<&serde_json::Map<String, Value>>(raw, "attention_other_setting", path)?.is_some() {
        let kind = text(raw, "attention_other_setting.attention_type", path)?;
        if kind != Some("sliding_attention") {
            refused!(
                Unimplemented,
                in raw,
                ["attention_other_setting"],
                "{} states attention_other_setting for {kind:?} layers; Ster reads it for sliding_attention layers",
                path.display()
            );
        }
        let width = whole(raw, "attention_other_setting.head_dim", path)?;
        if width.is_some_and(|width| width != architecture.head_dim) {
            refused!(
                Unimplemented,
                in raw,
                ["attention_other_setting", "head_dim"],
                "{} gives its sliding-window heads a head_dim other than {}; Ster builds both kinds one width",
                path.display(),
                architecture.head_dim
            );
        }
    }
    let scaled = raw
        .get("rope_scaling")
        .is_some_and(|scaling| !scaling.is_null());
    let full_only = list(raw, "yarn_only_types", path)?
        .is_some_and(|kinds| kinds.len() == 1 && kinds[0].as_str() == Some("full_attention"));
    if scaled && !full_only {
        refused!(
            Unimplemented,
            in raw,
            ["rope_scaling", "yarn_only_types"],
            "{} scales the rotation of sliding-window layers too (yarn_only_types); Ster scales the full-attention rotation only",
            path.display()
        );
    }
    architecture.sliding_layers = windowed;
    architecture.sliding_window = whole(raw, "sliding_window", path)?;
    architecture.norm_offset = true;
    if flag(raw, "use_qk_norm", path)? {
        architecture.query_key_norm = QueryKeyNorm::PerHead;
    }
    architecture.head_gate = flag(raw, "use_head_wise_attn_gate", path)?.then_some(GateFunction::Sigmoid);
    if let Some(rotated) = list(raw, "use_rope_layers", path)?
        .filter(|list| !list.is_empty())
    {
        if rotated.len() != layers {
            refused!(
                Contradictory,
                in raw,
                ["use_rope_layers", "num_hidden_layers"],
                "{} lists {} use_rope_layers entries for {layers} layers",
                path.display(),
                rotated.len()
            );
        }
        for (layer, entry) in rotated.iter().enumerate() {
            match entry.as_bool() {
                Some(true) => {}
                Some(false) => architecture.unrotated_layers |= 1u128 << layer,
                None => {
                    let key = format!("use_rope_layers[{layer}]");
                    return Err(mistyped(&key, entry, "true or false", path));
                }
            }
        }
    }
    let head_dim = architecture.head_dim;
    let (full_share, windowed_share) =
        per_kind(raw, "partial_rotary_factors", windowed, layers, path)?;
    // A kind whose share is unstated rotates its whole head.
    let width = |share: Option<f64>| match share {
        Some(share) => (head_dim as f64 * share) as usize,
        None => head_dim,
    };
    let (global_rotary, local_rotary) = (width(full_share), width(windowed_share));
    if [global_rotary, local_rotary]
        .iter()
        .any(|rotated| *rotated == 0 || rotated % 2 != 0 || *rotated > head_dim)
    {
        refused!(
            OutOfRange,
            in raw,
            ["head_dim", "partial_rotary_factors"],
            "{} rotates {global_rotary} and {local_rotary} of {head_dim} components per head (partial_rotary_factors); each must be even, above zero and at most the head",
            path.display()
        );
    }
    architecture.rotary_dim = local_rotary;
    let base = f64::from(llama.rope_theta);
    let (full_base, windowed_base) = per_kind(raw, "layer_rope_theta", windowed, layers, path)?;
    if full_base.is_some_and(|full| full != base) {
        refused!(
            Unimplemented,
            in raw,
            ["layer_rope_theta", "rope_theta"],
            "{} rotates its full-attention layers by more than one base; Ster rotates them all by one",
            path.display()
        );
    }
    // The sliding-window layers always rotate by their own table: its width
    // can differ from the full-attention layers' and it is never scaled.
    architecture.local_rope_theta = Some(match windowed_base {
        Some(windowed) => windowed,
        None => base,
    } as f32);
    architecture.global_attention = Some(GlobalAttention {
        heads: whole(raw, "num_global_attention_heads", path)?,
        head_dim,
        key_value_heads: whole(raw, "num_global_key_value_heads", path)?,
        rotary_dim: global_rotary,
        key_is_value: false,
    });
    let mut routed = step_experts(raw, layers, path)?;
    routed.scoring = match text(raw, "moe_router_activation", path)? {
        None | Some("sigmoid") => Scoring::Sigmoid,
        Some("softmax") => Scoring::Softmax,
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["moe_router_activation"],
            "{} declares moe_router_activation {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    routed.routed_scale = number(raw, "moe_router_scaling_factor", path)?;
    routed.selection_bias = flag(raw, "use_moe_router_bias", path)?.then_some("moe.router_bias");
    step_limits(
        raw,
        ("swiglu_limits", "swiglu_limits_shared"),
        layers,
        &mut routed,
        path,
    )?;
    architecture.experts = Some(routed);
    Ok(())
}
