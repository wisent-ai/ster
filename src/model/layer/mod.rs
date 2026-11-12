//! One decoder layer: its block — the mixer, the feed-forward, the norms
//! around both halves and how their outputs join the residual stream — and
//! what some families add around it.

#[path = "mixing/altup.rs"]
pub(super) mod altup;
mod block;
#[path = "mixing/dense.rs"]
mod dense;
pub(super) mod experts;
mod feed_forward;
#[path = "mixing/mixer.rs"]
mod mixer;
pub(super) mod norm;
mod recurrent;
mod run;
#[path = "mixing/shared.rs"]
pub(super) mod shared;

use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder, linear, linear_no_bias};

use super::{Activation, deepseek4::HyperConnection, depth::DepthMixes};
use block::Block;
use norm::Norm;

/// A projection, with a bias when the architecture says it carries one.
/// `conv1d` reads a weight stored `[inputs, outputs]`, as GPT-2's `Conv1D`
/// stores it, and lays it out `[outputs, inputs]` once at load.
pub(super) fn projection(
    inputs: usize,
    outputs: usize,
    bias: bool,
    conv1d: bool,
    builder: VarBuilder<'_>,
) -> candle_core::Result<Linear> {
    if conv1d {
        let weight = builder
            .get((inputs, outputs), "weight")?
            .t()?
            .contiguous()?;
        let bias = if bias {
            Some(builder.get(outputs, "bias")?)
        } else {
            None
        };
        return Ok(Linear::new(weight, bias));
    }
    if bias {
        linear(inputs, outputs, builder)
    } else {
        linear_no_bias(inputs, outputs, builder)
    }
}

/// One decoder layer: its block, and what Gemma 4 adds after it — the
/// per-layer input and the `layer_scalar` the whole output is multiplied by.
#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    block: Block,
    per_layer_input: Option<PerLayerInput>,
    scalar: Option<Tensor>,
    /// Gemma 3n's AltUp around the block.
    altup: Option<altup::AltUp>,
    /// Kimi-K3's attention residuals around the block.
    depth: Option<DepthMixes>,
    /// DeepSeek-V4's hyper-connections before attention and before the
    /// feed-forward.
    hyper: Option<(HyperConnection, HyperConnection)>,
}

/// What every layer reads beside the hidden state: the embeddings the first
/// layer received (Zamba2's hybrid layers) and the layer's own slice of
/// Gemma 4's per-layer inputs.
pub(super) struct LayerInputs<'a> {
    pub embedded: &'a Tensor,
    pub per_layer: Option<Tensor>,
}

/// Gemma 4's per-layer input: `per_layer_input_gate` and the activation over
/// the block's output, times the layer's input, through
/// `per_layer_projection` and `post_per_layer_input_norm`.
#[derive(Debug, Clone)]
struct PerLayerInput {
    gate: Linear,
    projection: Linear,
    norm: Norm,
    activation: Activation,
}
