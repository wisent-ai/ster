//! The rotation a config states, taken out of it before Candle parses the Llama keys.

pub(in crate::runtime::load::family) mod bases;
pub(in crate::runtime::load::family) mod head;
pub(in crate::runtime::load::family) mod scaling;

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::runtime::load::family::read::{number, number_or, stated, whole, within, within_or};

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
    let per_type = stated::<&serde_json::Map<String, Value>>(raw, "rope_parameters", path)?
        .filter(|parameters| parameters.contains_key("full_attention"))
        .map(|parameters| Value::Object(parameters.clone()));
    if let Some(per_type) = per_type {
        let sliding = stated::<&serde_json::Map<String, Value>>(
            raw,
            "rope_parameters.sliding_attention",
            path,
        )?
        .map(|sliding| Value::Object(sliding.clone()));
        if let Some(sliding) = sliding {
            let kind = scaling_kind(&sliding, "rope_parameters.sliding_attention", path)?;
            if !matches!(kind, "default" | "") {
                refused!(
                    Unimplemented,
                    in raw,
                    ["rope_parameters.sliding_attention"],
                    "{} scales its sliding_attention rotation ({kind:?}); Ster scales only the full-attention layers' rotation",
                    path.display()
                );
            }
            let theta = number(raw, "rope_parameters.sliding_attention.rope_theta", path)?;
            // The sliding-window layers' rotated share, which a family that
            // gives each kind its own (Laguna) reads.
            let share = number(
                raw,
                "rope_parameters.sliding_attention.partial_rotary_factor",
                path,
            )?;
            if let Some(object) = raw.as_object_mut() {
                if let Some(theta) = theta {
                    object
                        .entry("rope_local_base_freq")
                        .or_insert(Value::from(theta));
                }
                if let Some(share) = share {
                    object
                        .entry("local_partial_rotary_factor")
                        .or_insert(Value::from(share));
                }
            }
        }
        // Chunked layers (Rnj-1.5) rotate by the global table, so their
        // stated rotation must be the full-attention one.
        let chunked = stated::<&serde_json::Map<String, Value>>(
            raw,
            "rope_parameters.chunked_attention",
            path,
        )?;
        if chunked.is_some() && per_type.get("full_attention") != per_type.get("chunked_attention")
        {
            refused!(
                Unimplemented,
                in raw,
                ["rope_parameters.chunked_attention", "rope_parameters.full_attention"],
                "{} rotates its chunked_attention layers unlike its full_attention layers; Ster rotates both by the full-attention rotation",
                path.display()
            );
        }
        if let (Some(full), Some(object)) =
            (per_type.get("full_attention").cloned(), raw.as_object_mut())
        {
            object.insert("rope_parameters".to_owned(), full);
        }
    }
    if stated::<&serde_json::Map<String, Value>>(raw, "rope_parameters", path)?.is_some() {
        let parameters = raw["rope_parameters"].clone();
        let kind = scaling_kind(&parameters, "rope_parameters", path)?;
        let scaled = !matches!(kind, "default" | "");
        let theta = number(raw, "rope_parameters.rope_theta", path)?;
        // Transformers 5 states the rotated share of each head here too.
        let share = number(raw, "rope_parameters.partial_rotary_factor", path)?;
        if let Some(object) = raw.as_object_mut() {
            if let Some(theta) = theta {
                object.entry("rope_theta").or_insert(Value::from(theta));
            }
            if let Some(share) = share {
                object
                    .entry("partial_rotary_factor")
                    .or_insert(Value::from(share));
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
    let kind = scaling_kind(scaling, "rope_scaling", path)?.to_owned();
    let alpha = number(raw, "rope_scaling.alpha", path)?;
    match kind.as_str() {
        "llama3" => Ok(None),
        "default" | "linear" | "longrope" | "yarn" | "telechat3-yarn" | "proportional"
        | "deepseek_yarn" => Ok(raw
            .as_object_mut()
            .and_then(|object| object.remove("rope_scaling"))),
        // HunYuan states `dynamic` with an `alpha`: a fixed NTK-aware base,
        // `rope_theta * alpha^(d / (d - 2))` over the head width `d`, the
        // same at every length.
        "dynamic" => {
            // Without an alpha, dynamic NTK scaling changes the rotary base
            // only once a sequence outgrows `max_position_embeddings`, which
            // Ster never runs; within it this is the plain rotation.
            let Some(alpha) = alpha else {
                if let Some(object) = raw.as_object_mut() {
                    object.remove("rope_scaling");
                }
                return Ok(None);
            };
            let width = match whole(raw, "head_dim", path)? {
                Some(width) => Some(width),
                None => match (
                    whole(raw, "hidden_size", path)?,
                    whole(raw, "num_attention_heads", path)?,
                ) {
                    (Some(hidden), Some(heads)) => Some(hidden / heads.max(1)),
                    _ => None,
                },
            };
            let Some(width) = width.filter(|width| *width > 2) else {
                refused!(
                    Incomplete,
                    in raw,
                    ["rope_scaling", "head_dim", "hidden_size", "num_attention_heads"],
                    "{} declares dynamic rope_scaling with alpha but no head width to scale by",
                    path.display()
                );
            };
            let theta = number_or(raw, "rope_theta", DEFAULT_ROPE_THETA, path)?;
            let scaled = theta * alpha.powf(width as f64 / (width as f64 - 2.0));
            if let Some(object) = raw.as_object_mut() {
                object.remove("rope_scaling");
                object.insert("rope_theta".to_owned(), Value::from(scaled));
            }
            Ok(None)
        }
        other => refused!(
            Unimplemented,
            in raw,
            ["rope_scaling"],
            "{} declares rope_scaling {other:?}; Ster implements llama3, linear, longrope, yarn, deepseek_yarn, telechat3-yarn, proportional and dynamic rotary scaling",
            path.display()
        ),
    }
}

/// The rotary base a config without `rope_theta` uses, Llama's and Candle's
/// default.
pub(in crate::runtime::load::family) const DEFAULT_ROPE_THETA: f64 = 10_000.0;

/// The kind of rotation a `rope_scaling` or `rope_parameters` object states
/// as `rope_type` (or the older `type`); empty when it states neither.
/// `name` is where the object sits in the config, as a refusal names it.
pub(in crate::runtime::load::family) fn scaling_kind<'a>(
    scaling: &'a Value,
    name: &str,
    path: &Path,
) -> Result<&'a str> {
    match within(scaling, name, "rope_type", path)? {
        Some(kind) => Ok(kind),
        None => within_or(scaling, name, "type", "", path),
    }
}
