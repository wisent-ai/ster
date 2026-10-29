//! Zamba2.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, Names, ParameterNorm, Positions, SharedBlocksSpec, StateSpaceSpec,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, number, whole};
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
        "zamba2" => {
            // Zamba2: Mamba-2 layers, and on the layers `layers_block_type`
            // calls `hybrid` a shared transformer block (one of
            // `num_mem_blocks`, used in turn) whose output joins the Mamba-2
            // mixer's input. The block's attention reads the hidden state
            // beside the embeddings, `attention_hidden_size` wide, divides
            // its scores by `sqrt(attention_head_dim / 2)` and rotates only
            // under `use_mem_rope`.
            fits(layers, path)?;
            let Some(kinds) = raw.get("layers_block_type").and_then(Value::as_array) else {
                bail!(
                    "{} declares a Zamba2 model without layers_block_type",
                    path.display()
                );
            };
            if kinds.len() != layers {
                bail!(
                    "{} lists {} layers_block_type entries for {layers} layers",
                    path.display(),
                    kinds.len()
                );
            }
            let mut hybrid = 0u128;
            for (layer, kind) in kinds.iter().enumerate() {
                match kind.as_str() {
                    Some("hybrid") => hybrid |= 1u128 << layer,
                    Some("mamba") => {}
                    other => bail!(
                        "{} names layer {layer} {other:?}; a Zamba2 layer is mamba or hybrid",
                        path.display()
                    ),
                }
            }
            let mut heads =
                structured(raw, "n_mamba_heads", "mamba_headdim", "mamba_ngroups", path)?;
            // Transformers' `Zamba2MambaMixer` clamps the step to
            // `(time_step_min, inf)`.
            heads.step_limit = (number(raw, "time_step_min").unwrap_or(0.0), f64::INFINITY);
            let (Some(state), Some(kernel)) =
                (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
            else {
                bail!(
                    "{} declares a Zamba2 model without mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            let inner =
                (number(raw, "mamba_expand").unwrap_or(2.0) * llama.hidden_size as f64) as usize;
            if inner != heads.heads * heads.head_dim {
                bail!(
                    "{} sizes its Mamba-2 mixer at mamba_expand times the width, {inner}, but {} heads of {} make {}",
                    path.display(),
                    heads.heads,
                    heads.head_dim,
                    heads.heads * heads.head_dim
                );
            }
            let Some(blocks) = whole(raw, "num_mem_blocks").filter(|blocks| *blocks > 0) else {
                bail!(
                    "{} declares a Zamba2 model without num_mem_blocks",
                    path.display()
                );
            };
            let attention_adapters = flag(raw, "use_shared_attention_adapter");
            let feed_forward_adapters = flag(raw, "use_shared_mlp_adapter");
            let rank = whole(raw, "adapter_rank").unwrap_or(0);
            if (attention_adapters || feed_forward_adapters) && rank == 0 {
                bail!(
                    "{} turns on Zamba2's shared adapters without an adapter_rank",
                    path.display()
                );
            }
            architecture.names = Names::JAMBA;
            architecture.score_divisor = (architecture.head_dim as f64 / 2.0).sqrt();
            if !flag(raw, "use_mem_rope") {
                architecture.positions = Positions::None;
            }
            architecture.shared_blocks = Some(SharedBlocksSpec {
                hybrid_layers: hybrid,
                blocks,
                attention_input: whole(raw, "attention_hidden_size")
                    .unwrap_or(2 * llama.hidden_size),
                intermediate: llama.intermediate_size,
                rank,
                attention_adapters,
                feed_forward_adapters,
            });
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "add_bias_linear"),
                convolution_bias: raw
                    .get("use_conv_bias")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                layers: every_layer(layers, path)? & !hybrid,
                feed_forward: false,
                structured: Some(heads),
            });
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
