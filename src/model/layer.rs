//! One decoder block: the feed-forward half, the two norms around it, and
//! where a steering vector is added to the residual stream.

use candle_core::Tensor;
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear_no_bias, rms_norm};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::{
    Activation, Architecture, Cache, Mode, Pass, Route,
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
    activation: Activation,
}

impl FeedForward {
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let (hidden, intermediate) = (config.hidden_size, config.intermediate_size);
        // Phi-3 stores the gate and up projections as one `gate_up_proj`,
        // gate rows first; each is a row slice of that mapped weight.
        let (gate, up) = if architecture.fused_projections {
            let fused = builder.get((2 * intermediate, hidden), "gate_up_proj.weight")?;
            (
                Linear::new(fused.narrow(0, 0, intermediate)?, None),
                Linear::new(fused.narrow(0, intermediate, intermediate)?, None),
            )
        } else {
            (
                linear_no_bias(hidden, intermediate, builder.pp("gate_proj"))?,
                linear_no_bias(hidden, intermediate, builder.pp("up_proj"))?,
            )
        };
        Ok(Self {
            gate,
            up,
            down: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                builder.pp("down_proj"),
            )?,
            gate_adapter: adapters.get(layer, Target::Gate).cloned(),
            up_adapter: adapters.get(layer, Target::Up).cloned(),
            down_adapter: adapters.get(layer, Target::Down).cloned(),
            activation: architecture.activation,
        })
    }

    /// No `Pass` here — `silu` and the elementwise product both backpropagate,
    /// so the feed-forward block is already differentiable as written — but a
    /// `Route`, because its three projections are adapter sites like any other.
    pub(super) fn forward(&self, hidden: &Tensor, route: Route) -> candle_core::Result<Tensor> {
        let gate = project(&self.gate, self.gate_adapter.as_ref(), hidden, route)?;
        let gate = match self.activation {
            Activation::Silu => candle_nn::ops::silu(&gate)?,
            Activation::GeluTanh => gate.gelu()?,
        };
        let gated = (gate * project(&self.up, self.up_adapter.as_ref(), hidden, route)?)?;
        project(&self.down, self.down_adapter.as_ref(), &gated, route)
    }
}

#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    attention_norm: RmsNorm,
    attention: Attention,
    /// Gemma 2's norm over the attention output, before the residual add.
    attention_output_norm: Option<RmsNorm>,
    feed_forward_norm: RmsNorm,
    feed_forward: FeedForward,
    /// Gemma 2's norm over the feed-forward output, before the residual add.
    feed_forward_output_norm: Option<RmsNorm>,
}

impl DecoderLayer {
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let norm = |name: &str| {
            load_norm(
                config.hidden_size,
                config.rms_norm_eps,
                architecture.norm_offset,
                builder.pp(name),
            )
        };
        // Llama names the norm before the feed-forward
        // `post_attention_layernorm`; Gemma 2 uses that name for the norm
        // after attention and calls the one before the feed-forward
        // `pre_feedforward_layernorm`.
        let (attention_output_norm, feed_forward_norm, feed_forward_output_norm) =
            if architecture.sandwich_norms {
                (
                    Some(norm("post_attention_layernorm")?),
                    norm("pre_feedforward_layernorm")?,
                    Some(norm("post_feedforward_layernorm")?),
                )
            } else {
                (None, norm("post_attention_layernorm")?, None)
            };
        Ok(Self {
            attention_norm: norm("input_layernorm")?,
            attention: Attention::load(
                builder.pp("self_attn"),
                config,
                architecture,
                layer,
                adapters,
            )?,
            attention_output_norm,
            feed_forward_norm,
            feed_forward: FeedForward::load(
                builder.pp("mlp"),
                config,
                architecture,
                layer,
                adapters,
            )?,
            feed_forward_output_norm,
        })
    }

    /// The window this layer's attention looks through, if any.
    pub(super) fn window(&self) -> Option<usize> {
        self.attention.window()
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
        let attention = match &self.attention_output_norm {
            Some(norm) => normalize(norm, &attention, mode.pass)?,
            None => attention,
        };
        let hidden = (hidden + attention)?;
        let feed_forward = self.feed_forward.forward(
            &normalize(&self.feed_forward_norm, &hidden, mode.pass)?,
            mode.route,
        )?;
        let feed_forward = match &self.feed_forward_output_norm {
            Some(norm) => normalize(norm, &feed_forward, mode.pass)?,
            None => feed_forward,
        };
        hidden + feed_forward
    }
}

/// An RMS norm as the checkpoint stores it. Gemma scales by `1 + weight`
/// rather than `weight`, so its stored weights are shifted by one at load;
/// the shifted tensor is a new one, never a variable, so the base stays
/// frozen exactly as before.
pub(super) fn load_norm(
    size: usize,
    eps: f64,
    offset: bool,
    builder: VarBuilder<'_>,
) -> candle_core::Result<RmsNorm> {
    if offset {
        let weight = builder.get(size, "weight")?;
        Ok(RmsNorm::new((weight + 1.0)?, eps))
    } else {
        rms_norm(size, eps, builder)
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
