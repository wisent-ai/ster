//! One decoder block: the feed-forward half, the two norms around it, and
//! where a steering vector is added to the residual stream.

use candle_core::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear_no_bias, rms_norm};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::{
    Architecture, Cache, Mode, Pass, Route,
    attention::{Attention, project},
};

#[derive(Debug, Clone)]
pub(super) struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
    gate_adapter: Option<Adapter>,
    up_adapter: Option<Adapter>,
    down_adapter: Option<Adapter>,
}

impl FeedForward {
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            gate: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                builder.pp("gate_proj"),
            )?,
            up: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                builder.pp("up_proj"),
            )?,
            down: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                builder.pp("down_proj"),
            )?,
            gate_adapter: adapters.get(layer, Target::Gate).cloned(),
            up_adapter: adapters.get(layer, Target::Up).cloned(),
            down_adapter: adapters.get(layer, Target::Down).cloned(),
        })
    }

    /// No `Pass` here — `silu` and the elementwise product both backpropagate,
    /// so the feed-forward block is already differentiable as written — but a
    /// `Route`, because its three projections are adapter sites like any other.
    pub(super) fn forward(&self, hidden: &Tensor, route: Route) -> candle_core::Result<Tensor> {
        let gated = (candle_nn::ops::silu(&project(
            &self.gate,
            self.gate_adapter.as_ref(),
            hidden,
            route,
        )?)? * project(&self.up, self.up_adapter.as_ref(), hidden, route)?)?;
        project(&self.down, self.down_adapter.as_ref(), &gated, route)
    }
}

#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    attention_norm: RmsNorm,
    attention: Attention,
    feed_forward_norm: RmsNorm,
    feed_forward: FeedForward,
}

impl DecoderLayer {
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            attention_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("input_layernorm"),
            )?,
            attention: Attention::load(
                builder.pp("self_attn"),
                config,
                architecture,
                layer,
                adapters,
            )?,
            feed_forward_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("post_attention_layernorm"),
            )?,
            feed_forward: FeedForward::load(builder.pp("mlp"), config, layer, adapters)?,
        })
    }

    pub(super) fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let attention = self.attention.forward(
            &normalize(&self.attention_norm, hidden, mode.pass)?,
            index_pos,
            layer,
            cache,
            mask,
            mode,
        )?;
        let hidden = (hidden + attention)?;
        let feed_forward = self.feed_forward.forward(
            &normalize(&self.feed_forward_norm, &hidden, mode.pass)?,
            mode.route,
        )?;
        hidden + feed_forward
    }
}

/// RMS normalisation: fused for inference, composed for training.
///
/// `RmsNorm::forward` dispatches to `candle_nn::ops::rms_norm`, which ends in
/// `apply_op2_no_bwd` (candle-nn-0.11.0/src/ops.rs:684). `forward_diff`
/// (candle-nn-0.11.0/src/layer_norm.rs:197) is the same normalisation built
/// from `sqr`, `sum_keepdim`, `broadcast_div` and `broadcast_mul`, which do
/// record backward nodes.
pub(super) fn normalize(
    norm: &RmsNorm,
    hidden: &Tensor,
    pass: Pass,
) -> candle_core::Result<Tensor> {
    match pass {
        Pass::Inference => norm.forward(hidden),
        Pass::Differentiable => norm.forward_diff(hidden),
    }
}
