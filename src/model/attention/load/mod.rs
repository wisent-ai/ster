//! Building one layer's attention from the config's reading and the
//! checkpoint's weights: its widths, its query and key norms, its
//! projections, and the gates, biases and indexers its family adds.

mod fused;
mod parts;
mod qkv;

use candle_nn::VarBuilder;
use candle_transformers::models::llama::Config;

use crate::lora::{Adapters, Target};

use super::super::{
    Activation, Architecture, NormKind, Positions,
    layer::{experts::Experts, norm::NormSpec, projection},
};
use super::{Attention, Rotary, blocks};

/// The widths one layer's attention runs at.
#[derive(Debug, Clone, Copy)]
struct Shape {
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    query_width: usize,
    /// Width of one head's value, which latent attention makes narrower than
    /// the query.
    value_dim: usize,
    key_value_width: usize,
    value_width: usize,
}

impl Shape {
    /// Gemma 4's full-attention layers have their own head width, MiMo-V2's
    /// and Gemma 4's their own key-value head count, Step 3.5's their own
    /// head counts; DeciLM states each layer's key-value heads. Dots3-Note's
    /// sliding-window layers run their own latent attention: their own query
    /// heads, head width and value width.
    fn of(config: &Config, architecture: &Architecture, layer: usize) -> Self {
        let global = architecture.global_at(layer);
        let heads = match global.and_then(|global| global.heads) {
            Some(own) => own,
            None => config.num_attention_heads,
        };
        let own_key_value_heads = match global.and_then(|global| global.key_value_heads) {
            Some(own) => Some(own),
            None => architecture
                .layer_plan(layer)
                .and_then(|plan| plan.key_value_heads),
        };
        let key_value_heads = match own_key_value_heads {
            Some(own) => own,
            None => config.num_key_value_heads,
        };
        let (heads, head_dim, query_width, value_dim) = match (
            architecture.window_latent_at(layer),
            global,
        ) {
            (Some(own), _) => {
                let own_head = own.latent.unrotated + own.latent.rotated;
                (own.heads, own_head, own.heads * own_head, own.latent.value)
            }
            (None, Some(global)) => (
                heads,
                global.head_dim,
                heads * global.head_dim,
                match architecture.value_head_dim {
                    Some(value_head_dim) => value_head_dim,
                    None => global.head_dim,
                },
            ),
            (None, None) => (
                heads,
                architecture.head_dim,
                architecture.attention_width(heads),
                architecture.value_dim(),
            ),
        };
        Self {
            heads,
            key_value_heads,
            head_dim,
            query_width,
            value_dim,
            key_value_width: head_dim * key_value_heads,
            value_width: value_dim * key_value_heads,
        }
    }
}

impl Attention {
    /// `builder` is the layer's; the projections sit below the family's
    /// attention block (`self_attn`, `attention`, `attn`).
    pub(in crate::model) fn load(
        layer_builder: &VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        Self::load_reading(
            layer_builder,
            config,
            architecture,
            layer,
            adapters,
            config.hidden_size,
        )
    }

    /// [`Attention::load`] for an attention that reads `input` components
    /// rather than the model width (Zamba2's shared block reads the hidden
    /// state beside the embeddings, twice the width).
    pub(in crate::model) fn load_reading(
        layer_builder: &VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
        input: usize,
    ) -> candle_core::Result<Self> {
        let names = architecture.names;
        let builder = layer_builder.pp(names.attention);
        let window = architecture.window(layer);
        let shape = Shape::of(config, architecture, layer);
        let Shape {
            heads,
            key_value_heads,
            head_dim,
            value_dim,
            ..
        } = shape;
        let conv1d = architecture.conv1d;
        // A LayerNorm family's query and key norms carry no bias; the others
        // are RMS norms like the rest of the block. They work per head, never
        // in the hidden width's groups.
        let spec = NormSpec {
            kind: match architecture.norm {
                NormKind::Layer { .. } => NormKind::Layer { bias: false },
                other => other,
            },
            groups: 1,
            ..NormSpec::of(architecture)
        };
        let (query_norm, key_norm) =
            parts::query_key_norms(architecture, layer, spec, shape, &builder)?;
        let rotary = if !architecture.rotates(layer) {
            Rotary::None
        } else if window.is_some() && architecture.local_rope_theta.is_some() {
            Rotary::Local
        } else {
            Rotary::Global
        };
        // K2-Horizon's value experts, on its routed layers only.
        let value_experts = match &architecture.value_experts {
            Some(experts) if architecture.routed(layer) => Some(Experts::load(
                layer_builder,
                input,
                experts,
                Activation::Silu,
                layer,
            )?),
            _ => None,
        };
        let (projections, output_gate) = qkv::projections(
            &builder,
            architecture,
            layer,
            input,
            shape,
            value_experts.is_some(),
        )?;
        Ok(Self {
            projections,
            output: projection(
                heads * value_dim,
                config.hidden_size,
                architecture.output_bias,
                conv1d,
                layer_builder.pp(names.output),
            )?,
            query_adapter: adapters.get(layer, Target::Query).cloned(),
            key_adapter: adapters.get(layer, Target::Key).cloned(),
            value_adapter: adapters.get(layer, Target::Value).cloned(),
            output_adapter: adapters.get(layer, Target::Output).cloned(),
            query_norm,
            key_norm,
            query_key_norm: architecture.query_key_norm,
            norm_after_rotary: architecture.norm_after_rotary,
            heads,
            key_value_heads: if architecture.latent.is_some() {
                heads
            } else {
                key_value_heads
            },
            head_dim,
            value_dim,
            value_norm: if architecture.value_norm {
                Some(spec.unscaled(head_dim, &builder)?)
            } else {
                None
            },
            stores_key_values: architecture.stores_key_values(layer),
            index_shared: architecture
                .sparse_index
                .is_some_and(|index| index.shared(layer)),
            block_indexer: match architecture.block_index {
                Some(index) if index.indexes(layer) => {
                    Some(blocks::BlockIndexer::load(&builder, input, spec, index)?)
                }
                _ => None,
            },
            loop_mix: parts::loop_mix(layer_builder, architecture, layer, heads, head_dim)?,
            window,
            rotary,
            interleaved: architecture.interleaved_rotary,
            // A sliding-window latent attention scores over its own head.
            score_divisor: match architecture.window_latent_at(layer) {
                Some(_) => (head_dim as f64).sqrt(),
                None => architecture.score_divisor,
            },
            softcap: architecture.attention_softcap,
            prefix_bidirectional: architecture.prefix_lm,
            alibi: parts::alibi(architecture, heads, layer_builder.device())?,
            alibi_root: architecture.positions == Positions::AlibiRoot,
            clip_qkv: architecture.clip_qkv,
            key_scale: architecture.key_scale,
            value_scale: architecture.value_scale,
            query_temperature: architecture
                .query_temperature
                .filter(|temperature| temperature.at(layer)),
            output_gate,
            head_gate: match architecture.head_gate {
                Some(function) => Some((
                    projection(input, heads, false, conv1d, builder.pp("g_proj"))?,
                    function,
                )),
                None => None,
            },
            elementwise_gate: parts::elementwise_gate(&builder, architecture, input, shape)?,
            value_experts,
            low_rank: None,
            query_bottleneck: match architecture.query_bottleneck {
                Some(width) => Some((
                    projection(
                        input,
                        width,
                        architecture.query_key_value_bias,
                        conv1d,
                        builder.pp("q_proj"),
                    )?,
                    spec.load(width, builder.pp("inter_norm"))?,
                )),
                None => None,
            },
            sinks: parts::sinks(&builder, architecture, layer, heads)?,
        })
    }
}
