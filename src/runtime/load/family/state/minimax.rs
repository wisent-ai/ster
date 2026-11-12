//! MiniMax-Text-01 and MiniMax-M2.

use anyhow::Result;

use crate::model::{
    Architecture, ExpertLayout, LightningForm, LightningSpec, QueryKeyNorm, ScaledResiduals,
    Scoring,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{flag_or, list, number, number_or, text, whole_or};

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
        "minimax_text_01" | "minimax" => {
            // MiniMax-Text-01 (its own release spells `minimax_text_01`,
            // Transformers `minimax`): lightning attention on the layers
            // `attn_type_list` marks 0 (or `layer_types` calls
            // `linear_attention`), full attention over `rotary_dim` of each
            // head on the rest, Mixtral-named experts after every mixer, and
            // every sublayer joining the residual as `residual · alpha +
            // output · beta`, the residual being the normed input under
            // `postnorm`.
            fits(layers, path)?;
            let mut lightning = 0u128;
            if let Some(kinds) = list(raw, "attn_type_list", path)? {
                if kinds.len() != layers {
                    refused!(
                        Contradictory,
                        in raw,
                        ["attn_type_list", "num_hidden_layers"],
                        "{} lists {} attn_type_list entries for {layers} layers",
                        path.display(),
                        kinds.len()
                    );
                }
                for (layer, kind) in kinds.iter().enumerate() {
                    match kind.as_u64() {
                        Some(0) => lightning |= 1u128 << layer,
                        Some(1) => {}
                        _ => refused!(
                            Malformed,
                            in raw,
                            ["attn_type_list"],
                            "{} marks layer {layer} {kind} in attn_type_list; MiniMax layers are 0 (lightning) or 1 (full attention)",
                            path.display()
                        ),
                    }
                }
            } else if let Some(types) = list(raw, "layer_types", path)? {
                if types.len() != layers {
                    refused!(
                        Contradictory,
                        in raw,
                        ["layer_types", "num_hidden_layers"],
                        "{} lists {} layer_types for {layers} layers",
                        path.display(),
                        types.len()
                    );
                }
                for (layer, kind) in types.iter().enumerate() {
                    match kind.as_str() {
                        Some("linear_attention") => lightning |= 1u128 << layer,
                        Some("full_attention") => {}
                        other => refused!(
                            Malformed,
                            in raw,
                            ["layer_types"],
                            "{} names layer {layer} {other:?}; a MiniMax layer is linear_attention or full_attention",
                            path.display()
                        ),
                    }
                }
            } else {
                refused!(
                    Incomplete,
                    in raw,
                    ["attn_type_list", "layer_types"],
                    "{} declares a MiniMax model with neither attn_type_list nor layer_types",
                    path.display()
                );
            }
            if whole_or(raw, "shared_intermediate_size", 0, path)? > 0 {
                refused!(
                    Unimplemented,
                    in raw,
                    ["shared_intermediate_size"],
                    "{} declares shared experts (shared_intermediate_size); Ster implements MiniMax-Text-01 with routed experts only",
                    path.display()
                );
            }
            // The release's keys first, Transformers' after; one when unstated.
            let read = |keys: [&str; 2]| -> anyhow::Result<f64> {
                match number(raw, keys[0], path)? {
                    Some(stated) => Ok(stated),
                    None => number_or(raw, keys[1], 1.0, path),
                }
            };
            let pair = |alpha: [&str; 2], beta: [&str; 2]| -> anyhow::Result<(f64, f64)> {
                Ok((read(alpha)?, read(beta)?))
            };
            architecture.lightning = Some(LightningSpec {
                heads: llama.num_attention_heads,
                head_dim: architecture.head_dim,
                layers: lightning,
                form: LightningForm::MiniMax,
            });
            architecture.scaled_residuals = Some(ScaledResiduals {
                from_normed: flag_or(raw, "postnorm", true, path)?,
                linear_attention: pair(
                    [
                        "layernorm_linear_attention_alpha",
                        "linear_attn_alpha_factor",
                    ],
                    ["layernorm_linear_attention_beta", "linear_attn_beta_factor"],
                )?,
                full_attention: pair(
                    ["layernorm_full_attention_alpha", "full_attn_alpha_factor"],
                    ["layernorm_full_attention_beta", "full_attn_beta_factor"],
                )?,
                feed_forward: pair(
                    ["layernorm_mlp_alpha", "mlp_alpha_factor"],
                    ["layernorm_mlp_beta", "mlp_beta_factor"],
                )?,
            });
            architecture.experts = Some(experts(
                raw,
                "num_local_experts",
                "intermediate_size",
                true,
                ExpertLayout::Mixtral,
                0,
                path,
            )?);
        }
        "minimax_m2" => {
            // MiniMax-M2: full attention on every layer with query and key
            // norms over the whole projection (`qk_norm_type` per_layer) or
            // per head, rotation over `rotary_dim` of each head, and a
            // mixture of `num_local_experts` Mixtral-named experts scored by
            // sigmoid, chosen with `block_sparse_moe.e_score_correction_bias`
            // added (`use_routing_bias`) and renormalised.
            let linear_layers = list(raw, "attn_type_list", path)?
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_u64() == Some(0)));
            if linear_layers || whole_or(raw, "shared_intermediate_size", 0, path)? > 0 {
                refused!(
                    Unimplemented,
                    in raw,
                    ["attn_type_list", "shared_intermediate_size"],
                    "{} declares lightning-attention layers (attn_type_list 0) or shared experts (shared_intermediate_size); Ster implements MiniMax-M2 with full attention everywhere and routed experts only",
                    path.display()
                );
            }
            // Transformers' MiniMax-M2 always norms query and key; MiniMax's
            // own configs say so with `use_qk_norm`, and only `false` turns
            // it off.
            if flag_or(raw, "use_qk_norm", true, path)? {
                architecture.query_key_norm = match text(raw, "qk_norm_type", path)? {
                    None | Some("per_layer") => QueryKeyNorm::Full,
                    Some("per_head") => QueryKeyNorm::PerHead,
                    Some(other) => refused!(
                        Unimplemented,
                        in raw,
                        ["qk_norm_type"],
                        "{} declares qk_norm_type {other:?}; Ster implements per_layer and per_head",
                        path.display()
                    ),
                };
            }
            let mut routed = experts(
                raw,
                "num_local_experts",
                "intermediate_size",
                true,
                ExpertLayout::Mixtral,
                0,
                path,
            )?;
            routed.scoring = match text(raw, "scoring_func", path)? {
                None | Some("sigmoid") => Scoring::Sigmoid,
                Some("softmax") => Scoring::Softmax,
                Some(other) => refused!(
                    Unimplemented,
                    in raw,
                    ["scoring_func"],
                    "{} declares scoring_func {other:?}; Ster implements softmax and sigmoid expert scores",
                    path.display()
                ),
            };
            routed.selection_bias = flag_or(raw, "use_routing_bias", true, path)?
                .then_some("block_sparse_moe.e_score_correction_bias");
            architecture.experts = Some(routed);
        }
        other => refused!(
            Unimplemented,
            in raw,
            ["model_type"],
            "model architecture {other:?} has no decoder in this Ster build"
        ),
    }
    Ok(())
}
