//! Bamba and Nemotron-H.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{
    Architecture, ExpertLayout, FeedForwardKind, LatentExperts, Names, ParameterNorm, Positions,
    SharedExpert, SharedForm, StateSpaceSpec,
};
use crate::runtime::load::family::At;
use crate::runtime::load::family::experts::deepseek::deepseek_experts;
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, text, whole};
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
        layers,
        path,
        ..
    } = *at;
    match model_type {
        "bamba" => {
            // Bamba: Mamba-2 mixers with attention on the layers
            // `attn_layer_indices` lists, a feed-forward after every mixer,
            // and rotation over `attn_rotary_emb` components of each head.
            fits(layers, path)?;
            let attention = raw
                .get("attn_layer_indices")
                .and_then(Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(Value::as_u64)
                        .filter(|layer| (*layer as usize) < layers)
                        .fold(0u128, |set, layer| set | (1u128 << layer))
                })
                .unwrap_or(0);
            let heads = structured(raw, "mamba_n_heads", "mamba_d_head", "mamba_n_groups", path)?;
            let (Some(state), Some(kernel)) =
                (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
            else {
                bail!(
                    "{} declares a Bamba model without mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            if let Some(width) = whole(raw, "attn_rotary_emb") {
                architecture.rotary_dim = width.min(architecture.head_dim);
            }
            architecture.names = Names::JAMBA;
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
                layers: every_layer(layers, path)? & !attention,
                feed_forward: true,
                structured: Some(heads),
            });
        }
        "nemotron_h" | "nemotron_h_puzzle" => {
            // Nemotron-H: one norm and one sublayer per layer, its kind the
            // layer's letter in `hybrid_override_pattern` — `M` a Mamba-2
            // scan, `*` attention without rotation, `-` a squared-ReLU
            // `up_proj`/`down_proj` feed-forward, `E` a mixture of such
            // feed-forwards behind DeepSeek-V3's router.
            fits(layers, path)?;
            let Some(pattern) = text(raw, "hybrid_override_pattern") else {
                bail!(
                    "{} declares a Nemotron-H model without hybrid_override_pattern",
                    path.display()
                );
            };
            if pattern.chars().count() != layers {
                bail!(
                    "{} spells {} layers in hybrid_override_pattern for {layers} layers",
                    path.display(),
                    pattern.chars().count()
                );
            }
            let (mut mamba, mut feed_forward, mut routed_layers) = (0u128, 0u128, 0u128);
            for (layer, kind) in pattern.chars().enumerate() {
                match kind {
                    'M' => mamba |= 1u128 << layer,
                    '-' => feed_forward |= 1u128 << layer,
                    'E' => routed_layers |= 1u128 << layer,
                    '*' => {}
                    other => bail!(
                        "{} marks layer {layer} {other:?} in hybrid_override_pattern; Ster implements Nemotron-H's M (Mamba-2), * (attention), - (feed-forward) and E (mixture-of-experts) layers",
                        path.display()
                    ),
                }
            }
            if routed_layers != 0 {
                // `E` layers: `n_routed_experts` experts under `mixer.gate`
                // (sigmoid scores, `mixer.gate.e_score_correction_bias` added
                // to choose, `n_group`/`topk_group` limits, renormalised under
                // `norm_topk_prob`, scaled by `routed_scaling_factor`) beside
                // `n_shared_experts` shared ones under `mixer.shared_experts`,
                // `moe_shared_expert_intermediate_size` wide each; every
                // projection is `up_proj`/`down_proj`. Under `moe_latent_size`
                // the routed experts work on `mixer.fc1_latent_proj(x)` and
                // return through `mixer.fc2_latent_proj`. Nemotron Puzzle's
                // `block_configs` give each `moe` layer its own
                // `num_experts_per_tok` and `moe_intermediate_size`.
                let mut routed = deepseek_experts(raw, model_type, layers, path)?;
                routed.layout = ExpertLayout::NemotronH;
                routed.dense_layers = every_layer(layers, path)? & !routed_layers;
                routed.selection_bias = Some("mixer.gate.e_score_correction_bias");
                routed.latent = whole(raw, "moe_latent_size")
                    .filter(|width| *width > 0)
                    .map(|width| LatentExperts {
                        width,
                        down: LatentExperts::NEMOTRON_H_DOWN,
                        up: LatentExperts::NEMOTRON_H_UP,
                        norm: None,
                    });
                let shared_count = whole(raw, "n_shared_experts").unwrap_or(1).max(1);
                routed.shared = whole(raw, "moe_shared_expert_intermediate_size")
                    .filter(|width| *width > 0)
                    .map(|intermediate| SharedExpert {
                        intermediate: shared_count * intermediate,
                        module: "mixer.shared_experts",
                        gated: false,
                        form: SharedForm::UpDown,
                    });
                if let Some(blocks) = raw.get("block_configs").and_then(Value::as_array) {
                    if blocks.len() != layers {
                        bail!(
                            "{} lists {} block_configs for {layers} layers",
                            path.display(),
                            blocks.len()
                        );
                    }
                    architecture.expert_overrides = Some(
                        blocks
                            .iter()
                            .map(|block| {
                                let moe = block.get("block_type").and_then(Value::as_str) == Some("moe");
                                let top_k = whole(block, "num_experts_per_tok").unwrap_or(routed.top_k);
                                let width = whole(block, "moe_intermediate_size").unwrap_or(routed.intermediate);
                                if moe && (top_k == 0 || top_k > routed.count) {
                                    bail!(
                                        "{} routes a block's tokens to {top_k} of {} experts; it must be at least one and at most all of them",
                                        path.display(),
                                        routed.count
                                    );
                                }
                                Ok(moe.then_some((top_k, width)))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                architecture.experts = Some(routed);
            }
            let heads = structured(raw, "mamba_num_heads", "mamba_head_dim", "n_groups", path)?;
            let (Some(state), Some(kernel)) =
                (whole(raw, "ssm_state_size"), whole(raw, "conv_kernel"))
            else {
                bail!(
                    "{} declares a Nemotron-H model without ssm_state_size and conv_kernel",
                    path.display()
                );
            };
            architecture.names = if model_type == "nemotron_h_puzzle" {
                Names::NEMOTRON_H_MODEL
            } else {
                Names::NEMOTRON_H
            };
            architecture.positions = Positions::None;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.lone_sublayers = Some(feed_forward | routed_layers);
            architecture.state_space = Some(StateSpaceSpec {
                inner: heads.heads * heads.head_dim,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "use_bias"),
                convolution_bias: raw
                    .get("use_conv_bias")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                layers: mamba,
                feed_forward: false,
                structured: Some(heads),
            });
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}
