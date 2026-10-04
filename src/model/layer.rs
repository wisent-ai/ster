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
        architecture: &Architecture,
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
    /// The norm before attention; OLMo 2 has none.
    attention_norm: Option<RmsNorm>,
    attention: Attention,
    /// The norm over attention's output before the residual add (Gemma 2 and
    /// 3, OLMo 2).
    attention_output_norm: Option<RmsNorm>,
    /// The norm before the feed-forward; OLMo 2 has none.
    feed_forward_norm: Option<RmsNorm>,
    feed_forward: FeedForward,
    /// The norm over the feed-forward's output before the residual add.
    feed_forward_output_norm: Option<RmsNorm>,
    /// Granite's `residual_multiplier` on each sublayer's output.
    residual_multiplier: Option<f64>,
}

impl DecoderLayer {
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
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
        // The checkpoints name the same position differently:
        //
        // * Llama: `input_layernorm` before attention and
        //   `post_attention_layernorm` before the feed-forward.
        // * Gemma 2 and 3: those plus `post_attention_layernorm` over the
        //   attention output, so the norm before the feed-forward is
        //   `pre_feedforward_layernorm`, and `post_feedforward_layernorm`
        //   closes the block.
        // * OLMo 2: no norm before either sublayer; `post_attention_layernorm`
        //   and `post_feedforward_layernorm` sit over their outputs.
        let (attention_norm, attention_output_norm, feed_forward_norm, feed_forward_output_norm) =
            match (architecture.pre_norms, architecture.output_norms) {
                (true, false) => (
                    Some(norm("input_layernorm")?),
                    None,
                    Some(norm("post_attention_layernorm")?),
                    None,
                ),
                (true, true) => (
                    Some(norm("input_layernorm")?),
                    Some(norm("post_attention_layernorm")?),
                    Some(norm("pre_feedforward_layernorm")?),
                    Some(norm("post_feedforward_layernorm")?),
                ),
                (false, _) => (
                    None,
                    Some(norm("post_attention_layernorm")?),
                    None,
                    Some(norm("post_feedforward_layernorm")?),
                ),
            };
        Ok(Self {
            attention_norm,
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
            residual_multiplier: architecture.residual_multiplier,
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
            &optional_norm(self.attention_norm.as_ref(), hidden, mode.pass)?,
            index_pos,
            layer,
            cache,
            mask,
            mode,
        )?;
        let attention = optional_norm(self.attention_output_norm.as_ref(), &attention, mode.pass)?;
        let hidden = (hidden + self.scaled(attention)?)?;
        let feed_forward = self.feed_forward.forward(
            &optional_norm(self.feed_forward_norm.as_ref(), &hidden, mode.pass)?,
            mode.route,
        )?;
        let feed_forward =
            optional_norm(self.feed_forward_output_norm.as_ref(), &feed_forward, mode.pass)?;
        hidden + self.scaled(feed_forward)?
    }

    /// A sublayer's output as it joins the residual stream.
    fn scaled(&self, output: Tensor) -> candle_core::Result<Tensor> {
        match self.residual_multiplier {
            Some(multiplier) => output * multiplier,
            None => Ok(output),
        }
    }
}

/// `hidden` through `norm` when the block has one at this position.
fn optional_norm(
    norm: Option<&RmsNorm>,
    hidden: &Tensor,
    pass: Pass,
) -> candle_core::Result<Tensor> {
    match norm {
        Some(norm) => normalize(norm, hidden, pass),
        None => Ok(hidden.clone()),
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
