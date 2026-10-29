//! Inkling.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::model::{Architecture, InklingSpec, Names, Positions, RelativeHeads};
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{number, text, whole};

/// Inkling (`inkling_text`, inside `inkling_mm_model` below `model.llm`):
/// see [`InklingSpec`]. The sliding layers are `layer_types`'
/// `hybrid_sliding` entries, else `local_layer_ids`; the routed ones
/// `mlp_layer_types`' `sparse` entries, else every layer from
/// `dense_mlp_idx`. The embedding is RMS-normed by `embed_norm`, the head
/// reads the final norm's output divided by `logits_mup_width_multiplier`,
/// and only the first `unpadded_vocab_size` logits are scored. Every switch
/// Transformers' Inkling implements one way only — `use_sconv`,
/// `use_embed_norm`, `gate_activation` `sigmoid`, `use_gate_bias`,
/// `norm_after_topk`, `shared_expert_sink`, `use_global_scale`, no `q_bias`
/// or `o_bias`, no `final_logit_softcapping` — is refused set the other way.
pub(in crate::runtime::load::family) fn inkling(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    let size = |key: &str| -> Result<usize> {
        whole(raw, key)
            .filter(|size| *size > 0)
            .with_context(|| format!("{} declares an Inkling model without {key}", path.display()))
    };
    for (key, required) in [
        ("use_sconv", true),
        ("use_embed_norm", true),
        ("use_gate_bias", true),
        ("norm_after_topk", true),
        ("shared_expert_sink", true),
        ("use_global_scale", true),
        ("q_bias", false),
        ("o_bias", false),
    ] {
        if let Some(stated) = raw.get(key).and_then(Value::as_bool) {
            if stated != required {
                bail!(
                    "{} declares {key} {stated}; Ster implements Inkling with {key} {required}",
                    path.display()
                );
            }
        }
    }
    if let Some(activation) =
        text(raw, "gate_activation").filter(|activation| *activation != "sigmoid")
    {
        bail!(
            "{} declares gate_activation {activation:?}; Ster implements Inkling's sigmoid router",
            path.display()
        );
    }
    if raw
        .get("final_logit_softcapping")
        .is_some_and(|cap| !cap.is_null())
    {
        bail!(
            "{} declares final_logit_softcapping; Ster implements Inkling without it",
            path.display()
        );
    }
    let listed = |key: &str, chosen: &str| -> Result<Option<u128>> {
        let Some(entries) = raw.get(key).and_then(Value::as_array) else {
            return Ok(None);
        };
        if entries.len() != layers {
            bail!(
                "{} lists {} {key} entries for {layers} layers",
                path.display(),
                entries.len()
            );
        }
        Ok(Some(
            entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.as_str() == Some(chosen))
                .fold(0u128, |set, (layer, _)| set | 1u128 << layer),
        ))
    };
    let sliding_layers = match listed("layer_types", "hybrid_sliding")? {
        Some(set) => set,
        None => {
            let Some(local) = raw.get("local_layer_ids").and_then(Value::as_array) else {
                bail!(
                    "{} declares an Inkling model with neither layer_types nor local_layer_ids",
                    path.display()
                );
            };
            let mut set = 0u128;
            for entry in local {
                match entry.as_u64().map(|layer| layer as usize) {
                    Some(layer) if layer < layers => set |= 1u128 << layer,
                    _ => bail!(
                        "{} lists local_layer_ids entry {entry}, outside layers 0 to {}",
                        path.display(),
                        layers - 1
                    ),
                }
            }
            set
        }
    };
    let routed_layers = match listed("mlp_layer_types", "sparse")? {
        Some(set) => set,
        None => {
            let dense = whole(raw, "dense_mlp_idx").unwrap_or(0).min(layers);
            (dense..layers).fold(0u128, |set, layer| set | 1u128 << layer)
        }
    };
    let log_scaling = match number(raw, "log_scaling_n_floor") {
        Some(floor) if floor > 0.0 => {
            let Some(alpha) = number(raw, "log_scaling_alpha") else {
                bail!(
                    "{} declares log_scaling_n_floor without log_scaling_alpha",
                    path.display()
                );
            };
            Some((alpha, floor))
        }
        Some(floor) => bail!(
            "{} declares log_scaling_n_floor {floor}; it must be positive",
            path.display()
        ),
        None => None,
    };
    let Some(route_scale) = number(raw, "route_scale") else {
        bail!(
            "{} declares an Inkling model without route_scale",
            path.display()
        );
    };
    let spec = InklingSpec {
        full: RelativeHeads {
            heads: size("num_attention_heads")?,
            key_value_heads: size("num_key_value_heads")?,
            head_dim: size("head_dim")?,
            extent: size("rel_extent")?,
        },
        sliding: RelativeHeads {
            heads: size("swa_num_attention_heads")?,
            key_value_heads: size("swa_num_key_value_heads")?,
            head_dim: size("swa_head_dim")?,
            extent: size("sliding_window_size")?,
        },
        sliding_layers,
        profiles: size("d_rel")?,
        kernel: size("conv_kernel_size")?,
        log_scaling,
        routed_layers,
        experts: size("n_routed_experts")?,
        shared: size("n_shared_experts")?,
        top_k: size("num_experts_per_tok")?,
        expert_intermediate: size("moe_intermediate_size")?,
        route_scale,
    };
    for shape in [spec.full, spec.sliding] {
        if shape.heads % shape.key_value_heads != 0 {
            bail!(
                "{} shares {} key-value heads among {} heads; they must divide evenly",
                path.display(),
                shape.key_value_heads,
                shape.heads
            );
        }
    }
    if spec.top_k > spec.experts {
        bail!(
            "{} chooses {} of {} experts",
            path.display(),
            spec.top_k,
            spec.experts
        );
    }
    architecture.names = Names::INKLING;
    architecture.positions = Positions::None;
    architecture.embedding_norm = true;
    architecture.logits_multiplier = number(raw, "logits_mup_width_multiplier")
        .filter(|width| *width > 0.0)
        .map(|width| width.recip());
    architecture.vocabulary_limit = whole(raw, "unpadded_vocab_size");
    architecture.inkling = Some(spec);
    Ok(())
}
