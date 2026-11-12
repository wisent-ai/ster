//! Per-layer sets: one bit per layer, read from the lists and patterns configs state.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::list;

/// A layer set needs one bit per layer.
pub(in crate::runtime::load::family) fn fits(layers: usize, path: &Path) -> Result<()> {
    if layers > u128::BITS as usize {
        refused!(
            OutOfRange,
            ["num_hidden_layers" => layers],
            "{} has {layers} layers with per-layer attention settings; Ster tracks them per layer for at most {} layers",
            path.display(),
            u128::BITS
        );
    }
    Ok(())
}

/// SmolLM3's `no_rope_layers`: one entry per layer, `1` where the layer
/// rotates and `0` where it does not. Returns the layers that do not.
pub(in crate::runtime::load::family) fn unrotated(
    flags: &[Value],
    layers: usize,
    path: &Path,
) -> Result<u128> {
    if flags.len() != layers {
        refused!(
            Contradictory,
            ["no_rope_layers" => Value::from(flags), "num_hidden_layers" => layers],
            "{} lists {} no_rope_layers for {layers} layers",
            path.display(),
            flags.len()
        );
    }
    fits(layers, path)?;
    let mut set = 0u128;
    for (layer, flag) in flags.iter().enumerate() {
        match flag.as_u64() {
            Some(0) => set |= 1u128 << layer,
            Some(1) => {}
            _ => refused!(
                Malformed,
                ["no_rope_layers" => Value::from(flags)],
                "{} marks layer {layer} in no_rope_layers as {flag}; expected 0 or 1",
                path.display()
            ),
        }
    }
    Ok(set)
}

pub(in crate::runtime::load::family) fn every_layer(layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
    Ok((0..layers).fold(0, |set, layer| set | (1u128 << layer)))
}

pub(in crate::runtime::load::family) fn even_layers(layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
    Ok((0..layers)
        .step_by(2)
        .fold(0, |set, layer| set | (1u128 << layer)))
}

/// `layer_types`, as newer configs list it: one entry per layer, either
/// `sliding_attention` or `full_attention`.
pub(in crate::runtime::load::family) fn listed_layers(
    types: &[Value],
    layers: usize,
    path: &Path,
) -> Result<u128> {
    windowed_layers(types, layers, false, path)
}

/// [`listed_layers`], with `chunked_attention` entries windowed too when
/// the family attends by chunks (`chunked`).
pub(in crate::runtime::load::family) fn windowed_layers(
    types: &[Value],
    layers: usize,
    chunked: bool,
    path: &Path,
) -> Result<u128> {
    if types.len() != layers {
        refused!(
            Contradictory,
            ["layer_types" => Value::from(types), "num_hidden_layers" => layers],
            "{} lists {} layer_types for {layers} layers",
            path.display(),
            types.len()
        );
    }
    fits(layers, path)?;
    let mut set = 0u128;
    for (layer, kind) in types.iter().enumerate() {
        match kind.as_str() {
            Some("sliding_attention") => set |= 1u128 << layer,
            Some("chunked_attention") if chunked => set |= 1u128 << layer,
            Some("full_attention") => {}
            other => refused!(
                Unimplemented,
                ["layer_types" => Value::from(types)],
                "{} declares layer {layer} as {other:?}; Ster implements sliding_attention and full_attention{}",
                path.display(),
                if chunked {
                    " and chunked_attention"
                } else {
                    ""
                }
            ),
        }
    }
    Ok(set)
}

/// A per-layer list of ones and zeros under `key` (MiMo-V2's
/// `hybrid_layer_pattern` and `moe_layer_freq`): the layers marked one.
pub(in crate::runtime::load::family) fn flagged_layers(
    raw: &Value,
    key: &str,
    layers: usize,
    path: &Path,
) -> Result<u128> {
    let Some(flags) = list(raw, key, path)? else {
        refused!(Incomplete, in raw, [key], "{} lists no {key}", path.display());
    };
    if flags.len() != layers {
        refused!(
            Contradictory,
            [key => Value::from(flags), "num_hidden_layers" => layers],
            "{} lists {} {key} entries for {layers} layers",
            path.display(),
            flags.len()
        );
    }
    fits(layers, path)?;
    let mut set = 0u128;
    for (layer, flag) in flags.iter().enumerate() {
        match flag.as_u64() {
            Some(1) => set |= 1u128 << layer,
            Some(0) => {}
            _ => refused!(
                Malformed,
                in raw,
                [key],
                "{} marks layer {layer} in {key} as {flag}; Ster reads 0 or 1",
                path.display()
            ),
        }
    }
    Ok(set)
}
