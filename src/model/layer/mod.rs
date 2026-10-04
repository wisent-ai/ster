//! One decoder block: the feed-forward half, the norms around both halves,
//! and how their outputs join the residual stream.

mod experts;
pub(super) mod norm;

use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder, linear, linear_no_bias};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::{
    Activation, Architecture, Cache, FeedForwardKind, Mode, Pass, Route,
    attention::{Attention, project},
};
use experts::Experts;
use norm::{Norm, NormSpec};

/// The feed-forward half of a block: one dense feed-forward, or a router
/// over experts.
#[derive(Debug, Clone)]
enum FeedForwardBlock {
    Dense(FeedForward),
    Routed(Experts),
}

impl FeedForwardBlock {
    fn forward(&self, hidden: &Tensor, route: Route) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(dense) => dense.forward(hidden, route),
            Self::Routed(experts) => experts.forward(hidden),
        }
    }
}

/// A projection, with a bias when the architecture says it carries one.
pub(super) fn projection(
    inputs: usize,
    outputs: usize,
    bias: bool,
    builder: VarBuilder<'_>,
) -> candle_core::Result<Linear> {
    if bias {
        linear(inputs, outputs, builder)
    } else {
        linear_no_bias(inputs, outputs, builder)
    }
}

#[derive(Debug, Clone)]
pub(super) struct FeedForward {
    /// `None` on a plain feed-forward (Starcoder2, Phi-2, Nemotron).
    gate: Option<Linear>,
    up: Linear,
    down: Linear,
    gate_adapter: Option<Adapter>,
    up_adapter: Option<Adapter>,
    down_adapter: Option<Adapter>,
    activation: Activation,
}

impl FeedForward {
    /// `builder` is the layer's, so the family's names (`mlp.up_proj`,
    /// `mlp.c_fc`, `mlp.fc1`) are read from the architecture whole.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let (hidden, intermediate) = (config.hidden_size, config.intermediate_size);
        let bias = architecture.feed_forward_bias;
        let names = architecture.names;
        // Phi-3 and GLM store the gate and up projections as one matrix, gate
        // rows first; each is a row slice of that mapped weight.
        let (gate, up) = if architecture.fused_feed_forward {
            let fused = builder.pp(names.fused_gate_up);
            let weight = fused.get((2 * intermediate, hidden), "weight")?;
            let bias = if bias { Some(fused.get(2 * intermediate, "bias")?) } else { None };
            let slice = |start: usize| -> candle_core::Result<Linear> {
                Ok(Linear::new(
                    weight.narrow(0, start, intermediate)?,
                    bias.as_ref().map(|bias| bias.narrow(0, start, intermediate)).transpose()?,
                ))
            };
            (Some(slice(0)?), slice(intermediate)?)
        } else {
            let gate = match (architecture.feed_forward, names.gate) {
                (FeedForwardKind::Gated, Some(name)) => {
                    Some(projection(hidden, intermediate, bias, builder.pp(name))?)
                }
                _ => None,
            };
            (gate, projection(hidden, intermediate, bias, builder.pp(names.up))?)
        };
        Ok(Self {
            gate,
            up,
            down: projection(intermediate, hidden, bias, builder.pp(names.down))?,
            gate_adapter: adapters.get(layer, Target::Gate).cloned(),
            up_adapter: adapters.get(layer, Target::Up).cloned(),
            down_adapter: adapters.get(layer, Target::Down).cloned(),
            activation: architecture.activation,
        })
    }

    /// No `Pass` here — every activation and the elementwise product
    /// backpropagate, so the feed-forward block is already differentiable as
    /// written — but a `Route`, because its projections are adapter sites
    /// like any other.
    pub(super) fn forward(&self, hidden: &Tensor, route: Route) -> candle_core::Result<Tensor> {
        let up = project(&self.up, self.up_adapter.as_ref(), hidden, route)?;
        let inner = match &self.gate {
            Some(gate) => {
                let gate = project(gate, self.gate_adapter.as_ref(), hidden, route)?;
                (self.activation.apply(&gate)? * up)?
            }
            None => self.activation.apply(&up)?,
        };
        project(&self.down, self.down_adapter.as_ref(), &inner, route)
    }
}

#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    /// The norm before attention (before both halves in a parallel block);
    /// OLMo 2 has none.
    attention_norm: Option<Norm>,
    attention: Attention,
    /// The norm over attention's output before the residual add (Gemma 2 and
    /// 3, OLMo 2).
    attention_output_norm: Option<Norm>,
    /// The norm before the feed-forward; OLMo 2 and parallel blocks have none.
    feed_forward_norm: Option<Norm>,
    feed_forward: FeedForwardBlock,
    /// The norm over the feed-forward's output before the residual add.
    feed_forward_output_norm: Option<Norm>,
    /// Granite's `residual_multiplier` on each sublayer's output.
    residual_multiplier: Option<f64>,
    parallel: bool,
}

impl DecoderLayer {
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let spec = NormSpec::of(architecture);
        let norm = |name: &str| spec.load(config.hidden_size, builder.pp(name));
        let names = architecture.names;
        // Which norms a block has comes from the architecture; what each is
        // called comes from the family's names (Llama's
        // `post_attention_layernorm` before the feed-forward, Gemma 2's
        // `pre_feedforward_layernorm`, GLM-4's `post_self_attn_layernorm`
        // over attention's output). A parallel block (Cohere, Phi-2, GPT-J,
        // StableLM's `use_parallel_residual`) feeds one normalised input to
        // both halves, unless the feed-forward has its own norm (GPT-NeoX);
        // OLMo 2 has no norm before either sublayer.
        let (attention_norm, attention_output_norm, feed_forward_norm, feed_forward_output_norm) =
            match (architecture.parallel, architecture.pre_norms, architecture.output_norms) {
                (true, _, _) => (
                    Some(norm(names.attention_norm)?),
                    None,
                    if architecture.parallel_norms {
                        Some(norm(names.feed_forward_norm)?)
                    } else {
                        None
                    },
                    None,
                ),
                (false, true, false) => (
                    Some(norm(names.attention_norm)?),
                    None,
                    Some(norm(names.feed_forward_norm)?),
                    None,
                ),
                (false, true, true) => (
                    Some(norm(names.attention_norm)?),
                    Some(norm(names.attention_output_norm)?),
                    Some(norm(names.feed_forward_norm)?),
                    Some(norm(names.feed_forward_output_norm)?),
                ),
                (false, false, _) => (
                    None,
                    Some(norm(names.attention_output_norm)?),
                    None,
                    Some(norm(names.feed_forward_output_norm)?),
                ),
            };
        Ok(Self {
            attention_norm,
            attention: Attention::load(&builder, config, architecture, layer, adapters)?,
            attention_output_norm,
            feed_forward_norm,
            feed_forward: match &architecture.experts {
                Some(experts) if architecture.routed(layer) => FeedForwardBlock::Routed(
                    Experts::load(&builder, config.hidden_size, experts, architecture.activation)?,
                ),
                _ => FeedForwardBlock::Dense(FeedForward::load(
                    &builder,
                    config,
                    architecture,
                    layer,
                    adapters,
                )?),
            },
            feed_forward_output_norm,
            residual_multiplier: architecture.residual_multiplier,
            parallel: architecture.parallel,
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
        let normed = optional_norm(self.attention_norm.as_ref(), hidden, mode.pass)?;
        let attention = self
            .attention
            .forward(&normed, index_pos, layer, cache, mask, mode)?;
        let attention = optional_norm(self.attention_output_norm.as_ref(), &attention, mode.pass)?;
        if self.parallel {
            // GPT-NeoX normalises the feed-forward's input on its own; the
            // other parallel families reuse attention's.
            let feed_forward_input = match &self.feed_forward_norm {
                Some(norm) => norm.forward(hidden, mode.pass)?,
                None => normed,
            };
            let feed_forward = self.feed_forward.forward(&feed_forward_input, mode.route)?;
            return (hidden + self.scaled(attention)?)? + self.scaled(feed_forward)?;
        }
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
fn optional_norm(norm: Option<&Norm>, hidden: &Tensor, pass: Pass) -> candle_core::Result<Tensor> {
    match norm {
        Some(norm) => norm.forward(hidden, pass),
        None => Ok(hidden.clone()),
    }
}
