//! Zamba2's shared transformer blocks (`Zamba2AttentionDecoderLayer`).
//!
//! A block reads the hidden state beside the token embeddings, one vector
//! twice the model width, through `input_layernorm`; attends over it with
//! `self_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj`); passes the result
//! through `pre_ff_layernorm` and a fused gate-and-up `feed_forward`
//! (`gate_up_proj`, `down_proj`) with no residual add of its own. Each hybrid
//! layer reuses one of the blocks, adds its own low-rank terms to the
//! block's query, key, value and gate-and-up projections (slot k of the
//! block's `*_adapter_list`), and projects the block's output back through
//! its own `linear`; the result joins the input of the layer's Mamba-2
//! mixer.
//!
//! A block's weights are mapped once and shared, since each invocation's
//! copy holds the same tensor handles; only the low-rank terms and `linear`
//! are the invocation's own.

use candle_core::{D, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};
use candle_transformers::models::llama::Config;

use crate::lora::Adapters;
use crate::model::{
    Activation, Architecture, Cache, Mode, SharedBlocksSpec,
    attention::{Attention, LowRank},
};

use super::norm::{Norm, NormSpec};

#[derive(Debug, Clone)]
pub(in crate::model) struct SharedBlock {
    attention: Attention,
    attention_norm: Norm,
    feed_forward_norm: Norm,
    gate_up: Linear,
    down: Linear,
    activation: Activation,
    spec: SharedBlocksSpec,
}

impl SharedBlock {
    /// `builder` is the block's (`model.layers.{owner}.shared_transformer`);
    /// `owner` is the first hybrid layer that uses it.
    pub(in crate::model) fn load(
        builder: &VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        owner: usize,
        spec: SharedBlocksSpec,
    ) -> candle_core::Result<Self> {
        let norm = NormSpec::of(architecture);
        let hidden = config.hidden_size;
        Ok(Self {
            attention: Attention::load_reading(
                builder,
                config,
                architecture,
                owner,
                &Adapters::default(),
                spec.attention_input,
            )?,
            attention_norm: norm.load(spec.attention_input, builder.pp("input_layernorm"))?,
            feed_forward_norm: norm.load(hidden, builder.pp("pre_ff_layernorm"))?,
            gate_up: linear_no_bias(
                hidden,
                2 * spec.intermediate,
                builder.pp("feed_forward.gate_up_proj"),
            )?,
            down: linear_no_bias(
                spec.intermediate,
                hidden,
                builder.pp("feed_forward.down_proj"),
            )?,
            activation: architecture.activation,
            spec,
        })
    }
}

/// One hybrid layer's use of a shared block: the block, its own low-rank
/// terms on the gate-and-up projection, and its own output `linear`.
#[derive(Debug, Clone)]
pub(in crate::model) struct SharedInvocation {
    block: SharedBlock,
    gate_up_low_rank: Option<LowRank>,
    output: Linear,
}

impl SharedInvocation {
    /// `block_builder` is the shared block's, `layer_builder` the hybrid
    /// layer's (`model.layers.{layer}`); `slot` is how many hybrid layers
    /// come before this one.
    pub(in crate::model) fn load(
        block: &SharedBlock,
        block_builder: &VarBuilder<'_>,
        layer_builder: &VarBuilder<'_>,
        hidden: usize,
        slot: usize,
    ) -> candle_core::Result<Self> {
        let spec = block.spec;
        let low_rank =
            |list: VarBuilder<'_>, input: usize, output: usize| -> candle_core::Result<LowRank> {
                let entry = list.pp(slot.to_string());
                Ok(LowRank {
                    down: linear_no_bias(input, spec.rank, entry.pp("0"))?,
                    up: linear_no_bias(spec.rank, output, entry.pp("1"))?,
                })
            };
        let attention = block.attention.clone();
        let attention = if spec.attention_adapters {
            let projections = block_builder.pp("self_attn");
            let width = spec.attention_input;
            attention.with_low_rank([
                low_rank(projections.pp("linear_q_adapter_list"), width, width)?,
                low_rank(projections.pp("linear_k_adapter_list"), width, width)?,
                low_rank(projections.pp("linear_v_adapter_list"), width, width)?,
            ])
        } else {
            attention
        };
        let gate_up_low_rank = if spec.feed_forward_adapters {
            Some(low_rank(
                block_builder.pp("feed_forward.gate_up_proj_adapter_list"),
                hidden,
                2 * spec.intermediate,
            )?)
        } else {
            None
        };
        Ok(Self {
            block: SharedBlock {
                attention,
                ..block.clone()
            },
            gate_up_low_rank,
            output: linear_no_bias(hidden, hidden, layer_builder.pp("linear"))?,
        })
    }

    /// The block over `hidden` beside `embedded`, both `[batch, sequence,
    /// width]`, projected back by this layer's `linear`. The attention's keys
    /// and values are cached under `layer`.
    pub(in crate::model) fn forward(
        &self,
        hidden: &Tensor,
        embedded: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let block = &self.block;
        let joined = Tensor::cat(&[hidden, embedded], D::Minus1)?;
        let normed = block.attention_norm.forward(&joined, mode.pass)?;
        let attended = block
            .attention
            .forward(&normed, index_pos, layer, cache, mask, mode)?;
        let normed = block.feed_forward_norm.forward(&attended, mode.pass)?;
        let gate_up = block.gate_up.forward(&normed)?;
        let gate_up = match &self.gate_up_low_rank {
            Some(low_rank) => (gate_up + low_rank.forward(&normed)?)?,
            None => gate_up,
        };
        let intermediate = block.spec.intermediate;
        let gate = gate_up.narrow(D::Minus1, 0, intermediate)?;
        let up = gate_up.narrow(D::Minus1, intermediate, intermediate)?;
        let output = block.down.forward(&block.activation.gated(&gate, &up)?)?;
        self.output.forward(&output)
    }
}
