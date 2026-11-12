//! Gemma 4.

use anyhow::Result;

use crate::model::{
    Activation, Architecture, ExpertLayout, GlobalAttention, Names, PerLayerInputSpec, QueryKeyNorm,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::listed_layers;
use crate::runtime::load::family::read::{flag, list, number, text, whole, whole_or};
use crate::runtime::load::family::rope::scaling::rope_scaling;

/// What the families this file reads add to the Llama block; [`super::super::family`]
/// sends only their `model_type` values here.
pub(in crate::runtime::load::family) fn configure(
    at: &At,
    architecture: &mut Architecture,
) -> Result<()> {
    let At {
        model_type,
        raw,
        scaling,
        llama,
        layers,
        path,
    } = *at;
    match model_type {
        "gemma4_text" | "gemma4_unified_text" => {
            // Gemma 4: Gemma 3's norms around both sublayers and per-head
            // query and key norms, but norms that scale by their weight
            // itself, scores left undivided, a scale-free norm over each
            // head's value, and every block's output times its
            // `layer_scalar`. Its full-attention layers have wider heads
            // (`global_head_dim`), under `attention_k_eq_v` their own
            // key-value heads (`num_global_key_value_heads`) and values taken
            // from the key projection, and rotate by the `full_attention`
            // rotation over that width; the sliding-window layers rotate by
            // their own base.
            if text(raw, "use_bidirectional_attention", path)?
                == Some("all")
            {
                refused!(
                    Unimplemented,
                    in raw,
                    ["use_bidirectional_attention"],
                    "{} declares use_bidirectional_attention \"all\", so every token sees the whole sequence; Ster runs causal decoders only",
                    path.display()
                );
            }
            if flag(raw, "use_double_wide_mlp", path)? {
                refused!(
                    Unimplemented,
                    in raw,
                    ["use_double_wide_mlp"],
                    "{} declares use_double_wide_mlp, a doubled feed-forward on the key-value-sharing layers; Ster builds every Gemma 4 feed-forward intermediate_size wide",
                    path.display()
                );
            }
            let Some(global_head_dim) = whole(raw, "global_head_dim", path)? else {
                refused!(
                    Incomplete,
                    in raw,
                    ["global_head_dim"],
                    "{} declares a Gemma 4 model without global_head_dim",
                    path.display()
                );
            };
            let Some(types) = list(raw, "layer_types", path)? else {
                refused!(
                    Incomplete,
                    in raw,
                    ["layer_types"],
                    "{} declares a Gemma 4 model without layer_types",
                    path.display()
                );
            };
            architecture.local_rope_theta =
                number(raw, "rope_local_base_freq", path)?.map(|theta| theta as f32);
            if architecture.local_rope_theta.is_none() {
                refused!(
                    Incomplete,
                    in raw,
                    ["rope_local_base_freq", "rope_parameters"],
                    "{} declares a Gemma 4 model without a sliding_attention rotation base",
                    path.display()
                );
            }
            architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
            architecture.activation = Activation::GeluTanh;
            architecture.output_norms = true;
            architecture.names = Names::GEMMA2;
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.value_norm = true;
            architecture.score_divisor = 1.0;
            architecture.layer_scalar = true;
            architecture.final_softcap = number(raw, "final_logit_softcapping", path)?;
            architecture.sliding_window = whole(raw, "sliding_window", path)?;
            architecture.sliding_layers = listed_layers(types, layers, path)?;
            let key_is_value = flag(raw, "attention_k_eq_v", path)?;
            architecture.global_attention = Some(GlobalAttention {
                heads: None,
                head_dim: global_head_dim,
                // Transformers gives the full-attention layers their own
                // key-value head count only under `attention_k_eq_v`.
                key_value_heads: whole(raw, "num_global_key_value_heads", path)?.filter(|_| key_is_value),
                rotary_dim: global_head_dim,
                key_is_value,
            });
            architecture.rope_scaling = rope_scaling(scaling, global_head_dim, raw, llama, path)?;
            // The last `num_kv_shared_layers` layers reuse the keys and
            // values of the last earlier layer of their kind.
            let shared = whole_or(raw, "num_kv_shared_layers", 0, path)?;
            if shared > 0 {
                if shared >= layers {
                    refused!(
                        Contradictory,
                        in raw,
                        ["num_kv_shared_layers", "num_hidden_layers"],
                        "{} shares keys and values on {shared} of {layers} layers, leaving none to produce them",
                        path.display()
                    );
                }
                let first = layers - shared;
                architecture.shared_key_values = Some(first);
                if let Some(layer) =
                    (first..layers).find(|layer| architecture.key_value_source(*layer).is_none())
                {
                    refused!(
                        Contradictory,
                        in raw,
                        ["num_kv_shared_layers", "layer_types"],
                        "{} shares keys and values from layer {first} on, but layer {layer} has no earlier layer of its kind to share them from",
                        path.display()
                    );
                }
            }
            if let Some(width) =
                whole(raw, "hidden_size_per_layer_input", path)?.filter(|width| *width > 0)
            {
                architecture.per_layer_input = Some(PerLayerInputSpec {
                    width,
                    vocab: whole_or(raw, "vocab_size_per_layer_input", llama.vocab_size, path)?,
                });
            }
            if flag(raw, "enable_moe_block", path)? {
                architecture.side_experts = Some(experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    true,
                    ExpertLayout::Gemma4,
                    0,
                    path,
                )?);
            }
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
