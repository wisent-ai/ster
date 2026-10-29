//! The rotation a config states, taken out of it before Candle parses the Llama keys.

pub(in crate::runtime::load::family) mod bases;
pub(in crate::runtime::load::family) mod head;
pub(in crate::runtime::load::family) mod scaling;

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::runtime::load::family::read::{number, whole};

/// A `rope_scaling` Candle's Llama config cannot read (anything but Llama
/// 3's), taken out of the config before it is parsed so Ster can apply it
/// itself. Returns what was taken.
///
/// Configs written by Transformers 5 state the rotation in one
/// `rope_parameters` object; its `rope_theta` and `partial_rotary_factor` are
/// read as the base and the rotated share and, unless its `rope_type` is
/// `default`, the object as the scaling, never over keys the config states
/// at the top level. Gemma 3's and 4's state one object per
/// layer type: the `full_attention` one is the model's rotation, and the
/// `sliding_attention` one's base is `rope_local_base_freq`.
pub(in crate::runtime::load) fn take_rope_scaling(
    raw: &mut Value,
    path: &Path,
) -> Result<Option<Value>> {
    let per_type = raw
        .get("rope_parameters")
        .filter(|value| value.get("full_attention").is_some())
        .cloned();
    if let Some(per_type) = per_type {
        if let Some(sliding) = per_type
            .get("sliding_attention")
            .filter(|value| value.is_object())
        {
            let kind = scaling_kind(sliding);
            if !matches!(kind, "default" | "") {
                bail!(
                    "{} scales its sliding_attention rotation ({kind:?}); Ster scales only the full-attention layers' rotation",
                    path.display()
                );
            }
            if let (Some(theta), Some(object)) = (
                sliding
                    .get("rope_theta")
                    .filter(|theta| theta.is_number())
                    .cloned(),
                raw.as_object_mut(),
            ) {
                object.entry("rope_local_base_freq").or_insert(theta);
            }
            // The sliding-window layers' rotated share, which a family that
            // gives each kind its own (Laguna) reads.
            if let (Some(share), Some(object)) = (
                sliding
                    .get("partial_rotary_factor")
                    .filter(|share| share.is_number())
                    .cloned(),
                raw.as_object_mut(),
            ) {
                object.entry("local_partial_rotary_factor").or_insert(share);
            }
        }
        // Chunked layers (Rnj-1.5) rotate by the global table, so their
        // stated rotation must be the full-attention one.
        if let Some(chunked) = per_type
            .get("chunked_attention")
            .filter(|value| value.is_object())
        {
            if per_type.get("full_attention") != Some(chunked) {
                bail!(
                    "{} rotates its chunked_attention layers unlike its full_attention layers; Ster rotates both by the full-attention rotation",
                    path.display()
                );
            }
        }
        if let (Some(full), Some(object)) =
            (per_type.get("full_attention").cloned(), raw.as_object_mut())
        {
            object.insert("rope_parameters".to_owned(), full);
        }
    }
    if let Some(parameters) = raw
        .get("rope_parameters")
        .filter(|value| value.is_object())
        .cloned()
    {
        let scaled = !matches!(scaling_kind(&parameters), "default" | "");
        if let Some(object) = raw.as_object_mut() {
            if let Some(theta) = parameters
                .get("rope_theta")
                .filter(|theta| theta.is_number())
            {
                object.entry("rope_theta").or_insert_with(|| theta.clone());
            }
            // Transformers 5 states the rotated share of each head here too.
            if let Some(share) = parameters
                .get("partial_rotary_factor")
                .filter(|share| share.is_number())
            {
                object
                    .entry("partial_rotary_factor")
                    .or_insert_with(|| share.clone());
            }
            let unstated = object.get("rope_scaling").map_or(true, Value::is_null);
            if scaled && unstated {
                object.insert("rope_scaling".to_owned(), parameters);
            }
        }
    }
    let Some(scaling) = raw.get("rope_scaling").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    match scaling_kind(scaling) {
        "llama3" => Ok(None),
        "default" | "linear" | "longrope" | "yarn" | "telechat3-yarn" | "proportional"
        | "deepseek_yarn" => Ok(raw
            .as_object_mut()
            .and_then(|object| object.remove("rope_scaling"))),
        // HunYuan states `dynamic` with an `alpha`: a fixed NTK-aware base,
        // `rope_theta * alpha^(d / (d - 2))` over the head width `d`, the
        // same at every length.
        "dynamic" if scaling.get("alpha").and_then(Value::as_f64).is_some() => {
            let alpha = scaling.get("alpha").and_then(Value::as_f64).unwrap_or(1.0);
            let width = whole(raw, "head_dim").or_else(|| {
                Some(whole(raw, "hidden_size")? / whole(raw, "num_attention_heads")?.max(1))
            });
            let Some(width) = width.filter(|width| *width > 2) else {
                bail!(
                    "{} declares dynamic rope_scaling with alpha but no head width to scale by",
                    path.display()
                );
            };
            let theta = number(raw, "rope_theta").unwrap_or(DEFAULT_ROPE_THETA);
            let scaled = theta * alpha.powf(width as f64 / (width as f64 - 2.0));
            if let Some(object) = raw.as_object_mut() {
                object.remove("rope_scaling");
                object.insert("rope_theta".to_owned(), Value::from(scaled));
            }
            Ok(None)
        }
        // Without an alpha, dynamic NTK scaling changes the rotary base only
        // once a sequence outgrows `max_position_embeddings`, which Ster
        // never runs; within it this is the plain rotation.
        "dynamic" => {
            if let Some(object) = raw.as_object_mut() {
                object.remove("rope_scaling");
            }
            Ok(None)
        }
        other => bail!(
            "{} declares rope_scaling {other:?}; Ster implements llama3, linear, longrope, yarn, deepseek_yarn, telechat3-yarn, proportional and dynamic rotary scaling",
            path.display()
        ),
    }
}

/// The rotary base a config without `rope_theta` uses, Llama's and Candle's
/// default.
pub(in crate::runtime::load::family) const DEFAULT_ROPE_THETA: f64 = 10_000.0;

pub(in crate::runtime::load::family) fn scaling_kind(scaling: &Value) -> &str {
    scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("")
}
