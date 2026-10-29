//! Granite, GraniteMoE, its shared and sliding-window forms, Granite 4.0 and HyperCLOVAX.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, Names, ParameterNorm, Positions, SharedExpert, SharedForm,
    StateSpaceSpec,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, fits, listed_layers};
use crate::runtime::load::family::read::{flag, number, text, whole};
use crate::runtime::load::family::rope::bases::layer_bases;
use crate::runtime::load::family::state::mamba::structured;

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
        "granite" | "granitemoe" | "granitemoehybrid" | "granite_swa" | "granitemoe_swa"
        | "granitemoeshared" | "hyperclovax" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.embedding_multiplier = number(raw, "embedding_multiplier");
            architecture.residual_multiplier = number(raw, "residual_multiplier");
            if let Some(multiplier) = number(raw, "attention_multiplier") {
                architecture.score_divisor = 1.0 / multiplier;
            }
            // HyperCLOVAX multiplies its logits by `logits_scaling` where
            // Granite divides by it, and under `use_post_norm` (on unless
            // stated off) normalises each sublayer's output (`post_norm1`,
            // `post_norm2`) before the scaled residual add.
            architecture.logits_multiplier = if model_type == "hyperclovax" {
                number(raw, "logits_scaling")
            } else {
                number(raw, "logits_scaling").map(|scale| 1.0 / scale)
            };
            if model_type == "hyperclovax"
                && raw.get("use_post_norm").and_then(Value::as_bool) != Some(false)
            {
                architecture.output_norms = true;
                architecture.names = Names::HYPERCLOVAX;
            }
            if matches!(
                model_type,
                "granitemoe" | "granitemoe_swa" | "granitemoeshared"
            ) {
                // GraniteMoE takes the softmax over the top-k logits, which is
                // the full softmax renormalised over the chosen experts.
                let mut routed = experts(
                    raw,
                    "num_local_experts",
                    "intermediate_size",
                    true,
                    ExpertLayout::Granite,
                    0,
                    path,
                )?;
                // GraniteMoeShared and GraniteMoeSWA add `shared_mlp`,
                // `shared_intermediate_size` wide, to every layer's experts
                // when that width is above zero.
                if model_type != "granitemoe" {
                    routed.shared = whole(raw, "shared_intermediate_size")
                        .filter(|width| *width > 0)
                        .map(|intermediate| SharedExpert {
                            intermediate,
                            module: "shared_mlp",
                            gated: false,
                            form: SharedForm::Stacked,
                        });
                }
                architecture.experts = Some(routed);
            }
            if model_type.ends_with("_swa") {
                granite_windows(raw, layers, llama, architecture, path)?;
            }
            if model_type == "granitemoehybrid" {
                granite_hybrid(raw, layers, architecture, path)?;
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// Granite 4.0 (`granitemoehybrid`): Mamba-2 mixers (`mamba`, read from
/// Bamba's `mamba_*` keys) on the layers `layer_types` calls `mamba`,
/// attention on the others, rotation only when `position_embedding_type` is
/// `rope`, and after every mixer either GraniteMoE's experts beside the
/// stacked `shared_mlp` or, with no `num_local_experts`, the `shared_mlp`
/// alone as the dense feed-forward.
pub(in crate::runtime::load::family) fn granite_hybrid(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!(
            "{} declares a Granite 4.0 model without layer_types",
            path.display()
        );
    };
    if types.len() != layers {
        bail!(
            "{} lists {} layer_types for {layers} layers",
            path.display(),
            types.len()
        );
    }
    let mut mamba = 0u128;
    for (layer, kind) in types.iter().enumerate() {
        match kind.as_str() {
            Some("mamba") => mamba |= 1u128 << layer,
            Some("attention") => {}
            other => bail!(
                "{} names layer {layer} {other:?}; a Granite 4.0 layer is mamba or attention",
                path.display()
            ),
        }
    }
    architecture.positions = match text(raw, "position_embedding_type") {
        None | Some("rope") => Positions::Rotary,
        Some("nope") => Positions::None,
        Some(other) => bail!(
            "{} declares position_embedding_type {other:?}; Granite 4.0 rotates (rope) or does not (nope)",
            path.display()
        ),
    };
    let heads = structured(raw, "mamba_n_heads", "mamba_d_head", "mamba_n_groups", path)?;
    let (Some(state), Some(kernel)) = (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
    else {
        bail!(
            "{} declares a Granite 4.0 model without mamba_d_state and mamba_d_conv",
            path.display()
        );
    };
    architecture.state_space = Some(StateSpaceSpec {
        inner: heads.heads * heads.head_dim,
        state,
        kernel,
        step_rank: 0,
        projection_bias: flag(raw, "mamba_proj_bias"),
        convolution_bias: raw
            .get("mamba_conv_bias")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        parameter_norm: ParameterNorm::None,
        layers: mamba,
        feed_forward: true,
        structured: Some(heads),
    });
    let Some(shared) = whole(raw, "shared_intermediate_size") else {
        bail!(
            "{} declares a Granite 4.0 model without shared_intermediate_size",
            path.display()
        );
    };
    if whole(raw, "num_local_experts").unwrap_or(0) == 0 {
        architecture.names = Names::GRANITE_HYBRID;
        architecture.fused_feed_forward = true;
    } else {
        let mut routed = experts(
            raw,
            "num_local_experts",
            "intermediate_size",
            true,
            ExpertLayout::Granite,
            0,
            path,
        )?;
        routed.shared = Some(SharedExpert {
            intermediate: shared,
            module: "shared_mlp",
            gated: false,
            form: SharedForm::Stacked,
        });
        architecture.experts = Some(routed);
    }
    Ok(())
}

/// How often GraniteSWA's default layout attends fully: every fourth layer
/// from the first (`GraniteSWAConfig.__post_init__`, `i % 4 == 0`).
pub(in crate::runtime::load::family) const GRANITE_SWA_FULL_EVERY: usize = 4;

/// GraniteSWA's and GraniteMoeSWA's additions to Granite: a learned sink per
/// head (`self_attn.sinks`), sliding-window layers as `layer_types` lists
/// them (every layer but each fourth from the first without it), and
/// `layer_rope_theta`, a base per layer where zero means no rotation. Ster
/// rotates full-attention layers by `rope_theta` and sliding-window layers
/// by one base of their own, so the stated bases must fit that.
pub(in crate::runtime::load::family) fn granite_windows(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    architecture.attention_sinks = every_layer(layers, path)?;
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = match raw.get("layer_types").and_then(Value::as_array) {
        Some(types) => listed_layers(types, layers, path)?,
        None => {
            fits(layers, path)?;
            (0..layers)
                .filter(|layer| layer % GRANITE_SWA_FULL_EVERY != 0)
                .fold(0, |set, layer| set | (1u128 << layer))
        }
    };
    layer_bases(raw, layers, llama, architecture, path)
}
