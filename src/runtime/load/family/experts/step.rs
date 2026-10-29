//! Step's mixture of experts and its per-layer limits.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::model::{ExpertLayout, MixtureOfExperts, SharedExpert, SharedForm, SwigluLimit};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, text, whole};

/// Step's experts (Step3, Step 3.5): the layers `moe_layers_enum` lists
/// (from zero; all but the first when absent) route over `moe_num_experts`
/// experts stacked under `moe`, renormalised under `norm_expert_weight`,
/// beside a shared `share_expert` `share_expert_dim` wide.
pub(in crate::runtime::load::family) fn step_experts(
    raw: &Value,
    layers: usize,
    path: &Path,
) -> Result<MixtureOfExperts> {
    fits(layers, path)?;
    let routed_layers = match text(raw, "moe_layers_enum") {
        Some(list) => {
            let mut set = 0u128;
            for entry in list.split(',') {
                match entry.trim().parse::<usize>() {
                    Ok(layer) if layer < layers => set |= 1u128 << layer,
                    _ => bail!(
                        "{} lists moe_layers_enum entry {entry:?}, which names no layer below {layers}",
                        path.display()
                    ),
                }
            }
            set
        }
        None => every_layer(layers, path)? & !1u128,
    };
    let mut routed = experts(
        raw,
        "moe_num_experts",
        "moe_intermediate_size",
        flag(raw, "norm_expert_weight"),
        ExpertLayout::Step3,
        every_layer(layers, path)? & !routed_layers,
        path,
    )?;
    routed.shared = whole(raw, "share_expert_dim")
        .filter(|width| *width > 0)
        .map(|intermediate| SharedExpert {
            intermediate,
            module: "share_expert",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    Ok(routed)
}

/// Per-layer SwiGLU clamps on the routed and shared experts (Step 3.5's
/// `swiglu_limits` and `swiglu_limits_shared`, Ling 3.0's
/// `expert_swiglu_limit_list` and `share_expert_swiglu_limit_list`): one
/// entry per layer, zero or null meaning none. A dense layer's shared limit
/// is refused, since Ster clamps expert feed-forwards only.
pub(in crate::runtime::load::family) fn step_limits(
    raw: &Value,
    (routed_key, shared_key): (&str, &str),
    layers: usize,
    routed: &mut MixtureOfExperts,
    path: &Path,
) -> Result<()> {
    let limits = |key: &str| -> Result<Vec<f64>> {
        match raw.get(key).filter(|list| !list.is_null()) {
            None => Ok(Vec::new()),
            Some(list) => list
                .as_array()
                .and_then(|entries| {
                    entries
                        .iter()
                        .map(|entry| {
                            if entry.is_null() {
                                Some(0.0)
                            } else {
                                entry.as_f64()
                            }
                        })
                        .collect()
                })
                .with_context(|| {
                    format!(
                        "{} declares {key} that is not a list of numbers",
                        path.display()
                    )
                }),
        }
    };
    let (routed_limits, shared_limits) = (limits(routed_key)?, limits(shared_key)?);
    if let Some(layer) = (0..layers).find(|layer| {
        routed.dense_layers & (1u128 << layer) != 0
            && shared_limits.get(*layer).is_some_and(|limit| *limit != 0.0)
    }) {
        bail!(
            "{} clamps the dense feed-forward of layer {layer} ({shared_key}); Ster clamps expert feed-forwards only",
            path.display()
        );
    }
    if routed_limits
        .iter()
        .chain(&shared_limits)
        .any(|limit| *limit != 0.0)
    {
        routed.swiglu_limit = Some(SwigluLimit::Step {
            routed: routed_limits,
            shared: shared_limits,
        });
    }
    Ok(())
}
