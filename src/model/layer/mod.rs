//! One decoder block: the feed-forward half, the norms around both halves,
//! and how their outputs join the residual stream.

mod experts;
pub(super) mod norm;
mod recurrent;

use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder, linear, linear_no_bias};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::{
    Activation, Architecture, Cache, DeltaRuleForm, FeedForwardKind, Mode, ParallelScan, Pass, Route,
    attention::{Attention, project},
};
use experts::Experts;
use norm::{Norm, NormSpec};
use recurrent::{
    delta::DeltaRule,
    lightning::Lightning,
    state_space::{ShortConv, StateSpace},
    structured::Structured,
};

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
        let weight = builder.get((inputs, outputs), "weight")?.t()?.contiguous()?;
        let bias = if bias { Some(builder.get(outputs, "bias")?) } else { None };
        return Ok(Linear::new(weight, bias));
    }
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
    /// Falcon-H1's `mlp_multipliers`: on the gate before its activation,
    /// and on the output.
    scales: Option<(f64, f64)>,
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
        let conv1d = architecture.conv1d;
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
                    Some(projection(hidden, intermediate, bias, conv1d, builder.pp(name))?)
                }
                _ => None,
            };
            (gate, projection(hidden, intermediate, bias, conv1d, builder.pp(names.up))?)
        };
        Ok(Self {
            gate,
            up,
            down: projection(
                intermediate,
                hidden,
                bias || architecture.down_bias,
                conv1d,
                builder.pp(names.down),
            )?,
            gate_adapter: adapters.get(layer, Target::Gate).cloned(),
            up_adapter: adapters.get(layer, Target::Up).cloned(),
            down_adapter: adapters.get(layer, Target::Down).cloned(),
            activation: architecture.activation,
            scales: architecture.feed_forward_scales,
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
                let gate = match self.scales {
                    Some((gate_scale, _)) => (gate * gate_scale)?,
                    None => gate,
                };
                (self.activation.apply(&gate)? * up)?
            }
            None => self.activation.apply(&up)?,
        };
        let output = project(&self.down, self.down_adapter.as_ref(), &inner, route)?;
        match self.scales {
            Some((_, output_scale)) => output * output_scale,
            None => Ok(output),
        }
    }
}

/// What mixes information across positions in a block.
#[derive(Debug, Clone)]
enum Mixer {
    Attention(Attention),
    /// Mamba's selective state-space scan.
    StateSpace(StateSpace),
    /// Mamba-2's multi-head structured scan.
    Structured(Structured),
    /// LFM2's gated short convolution.
    ShortConv(ShortConv),
    /// Qwen3-Next's gated delta-rule linear attention.
    DeltaRule(DeltaRule),
    /// MiniMax-Text-01's lightning attention.
    Lightning(Lightning),
    /// Nemotron-H's feed-forward layers: the feed-forward is the block's
    /// one sublayer.
    FeedForward(FeedForwardBlock),
    /// Falcon-H1: attention and a Mamba-2 scan side by side.
    Parallel(Box<ParallelMixers>),
}

/// Falcon-H1's two mixers on one normed input, each output scaled before
/// the two are added.
#[derive(Debug, Clone)]
struct ParallelMixers {
    attention: Attention,
    scan: Structured,
    scales: ParallelScan,
}

impl ParallelMixers {
    fn forward(
        &self,
        normed: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let ParallelScan { attention_in, attention_out, scan_out } = self.scales;
        let scanned = (self.scan.forward(normed, layer, cache)? * scan_out)?;
        let attended =
            self.attention.forward(&(normed * attention_in)?, index_pos, layer, cache, mask, mode)?;
        scanned + (attended * attention_out)?
    }
}

#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    /// The norm before the mixer (before both halves in a parallel block);
    /// OLMo 2 has none.
    attention_norm: Option<Norm>,
    mixer: Mixer,
    /// The norm over attention's output before the residual add (Gemma 2 and
    /// 3, OLMo 2).
    attention_output_norm: Option<Norm>,
    /// The norm before the feed-forward; OLMo 2 and parallel blocks have none.
    feed_forward_norm: Option<Norm>,
    /// `None` in a state-space block, which is its mixer alone.
    feed_forward: Option<FeedForwardBlock>,
    /// The norm over the feed-forward's output before the residual add.
    feed_forward_output_norm: Option<Norm>,
    /// Granite's `residual_multiplier` on each sublayer's output.
    residual_multiplier: Option<f64>,
    parallel: bool,
    /// MiniMax-Text-01's scaled residuals for this block, if any.
    scales: Option<BlockScales>,
}

/// How a MiniMax-Text-01 block's sublayers join the residual stream:
/// `residual · alpha + output · beta`, the residual being the normed input
/// under `from_normed`.
#[derive(Debug, Clone, Copy)]
struct BlockScales {
    from_normed: bool,
    mixer: (f64, f64),
    feed_forward: (f64, f64),
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
        // MiniMax-Text-01's lightning-attention layers: a norm, the
        // lightning mixer, a norm and the feed-forward, each joining the
        // residual by the layer type's scales.
        if let Some(&lightning) = architecture.lightning_at(layer) {
            return Ok(Self {
                attention_norm: Some(norm(names.attention_norm)?),
                mixer: Mixer::Lightning(Lightning::load(
                    builder.pp(names.attention),
                    config.hidden_size,
                    layer,
                    config.num_hidden_layers,
                    lightning,
                )?),
                attention_output_norm: None,
                feed_forward_norm: Some(norm(names.feed_forward_norm)?),
                feed_forward: Some(feed_forward_block(&builder, config, architecture, layer, adapters)?),
                feed_forward_output_norm: None,
                residual_multiplier: architecture.residual_multiplier,
                parallel: false,
                scales: architecture.scaled_residuals.map(|scales| BlockScales {
                    from_normed: scales.from_normed,
                    mixer: scales.linear_attention,
                    feed_forward: scales.feed_forward,
                }),
            });
        }
        // Qwen3-Next's linear-attention layers: a norm, the delta-rule
        // mixer, a norm and the feed-forward, as in its attention layers.
        if let Some(&spec) = architecture.delta_rule_at(layer) {
            return Ok(Self {
                attention_norm: Some(norm(names.attention_norm)?),
                mixer: Mixer::DeltaRule(DeltaRule::load(
                    builder.pp(match spec.form {
                        DeltaRuleForm::Qwen3Next => "linear_attn",
                        DeltaRuleForm::Kimi => "self_attn",
                    }),
                    config.hidden_size,
                    config.rms_norm_eps,
                    spec,
                )?),
                attention_output_norm: None,
                feed_forward_norm: Some(norm(names.feed_forward_norm)?),
                feed_forward: Some(feed_forward_block(&builder, config, architecture, layer, adapters)?),
                feed_forward_output_norm: None,
                residual_multiplier: architecture.residual_multiplier,
                parallel: false,
                scales: None,
            });
        }
        // LFM2's convolution layers: a norm, the convolution, a norm and the
        // feed-forward, each half added to the residual.
        if let Some(convolution) = architecture.short_convolution_at(layer) {
            return Ok(Self {
                attention_norm: Some(norm(names.attention_norm)?),
                mixer: Mixer::ShortConv(ShortConv::load(
                    builder.pp(names.state_space),
                    config.hidden_size,
                    convolution,
                )?),
                attention_output_norm: None,
                feed_forward_norm: Some(norm(names.feed_forward_norm)?),
                feed_forward: Some(feed_forward_block(&builder, config, architecture, layer, adapters)?),
                feed_forward_output_norm: None,
                residual_multiplier: architecture.residual_multiplier,
                parallel: false,
                scales: None,
            });
        }
        // A state-space block is one norm, the mixer and the residual add —
        // the whole block in Mamba; in Jamba a norm and a feed-forward follow
        // as in any other block.
        if let Some(state_space) = architecture.state_space_at(layer) {
            let feed_forward = if state_space.feed_forward {
                Some(feed_forward_block(&builder, config, architecture, layer, adapters)?)
            } else {
                None
            };
            return Ok(Self {
                attention_norm: Some(norm(names.attention_norm)?),
                mixer: match state_space.structured {
                    Some(heads) => Mixer::Structured(Structured::load(
                        builder.pp(names.state_space),
                        config.hidden_size,
                        config.rms_norm_eps,
                        state_space,
                        heads,
                    )?),
                    None => Mixer::StateSpace(StateSpace::load(
                        builder.pp(names.state_space),
                        config.hidden_size,
                        config.rms_norm_eps,
                        state_space,
                    )?),
                },
                attention_output_norm: None,
                feed_forward_norm: if feed_forward.is_some() {
                    Some(norm(names.feed_forward_norm)?)
                } else {
                    None
                },
                feed_forward,
                feed_forward_output_norm: None,
                residual_multiplier: architecture.residual_multiplier,
                parallel: false,
                scales: None,
            });
        }
        // Falcon-H1: attention and a Mamba-2 scan side by side on one normed
        // input, added to the residual, then a norm and the feed-forward.
        if let (Some(scales), Some(state_space)) =
            (architecture.parallel_scan, architecture.state_space.as_ref())
        {
            let Some(heads) = state_space.structured else {
                candle_core::bail!("a parallel scan runs Mamba-2 heads, and this architecture states none");
            };
            return Ok(Self {
                attention_norm: Some(norm(names.attention_norm)?),
                mixer: Mixer::Parallel(Box::new(ParallelMixers {
                    attention: Attention::load(&builder, config, architecture, layer, adapters)?,
                    scan: Structured::load(
                        builder.pp(names.state_space),
                        config.hidden_size,
                        config.rms_norm_eps,
                        state_space,
                        heads,
                    )?,
                    scales,
                })),
                attention_output_norm: None,
                feed_forward_norm: Some(norm(names.feed_forward_norm)?),
                feed_forward: Some(feed_forward_block(&builder, config, architecture, layer, adapters)?),
                feed_forward_output_norm: None,
                residual_multiplier: architecture.residual_multiplier,
                parallel: false,
                scales: None,
            });
        }
        // Nemotron-H: one norm and one sublayer per block — the feed-forward
        // alone on the layers that name it, attention alone on the others.
        if let Some(feed_forward_layers) = architecture.lone_sublayers {
            let mixer = if layer < u128::BITS as usize && feed_forward_layers & (1u128 << layer) != 0 {
                Mixer::FeedForward(feed_forward_block(&builder, config, architecture, layer, adapters)?)
            } else {
                Mixer::Attention(Attention::load(&builder, config, architecture, layer, adapters)?)
            };
            return Ok(Self {
                attention_norm: Some(norm(names.attention_norm)?),
                mixer,
                attention_output_norm: None,
                feed_forward_norm: None,
                feed_forward: None,
                feed_forward_output_norm: None,
                residual_multiplier: architecture.residual_multiplier,
                parallel: false,
                scales: None,
            });
        }
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
            mixer: Mixer::Attention(Attention::load(&builder, config, architecture, layer, adapters)?),
            attention_output_norm,
            feed_forward_norm,
            feed_forward: Some(feed_forward_block(&builder, config, architecture, layer, adapters)?),
            feed_forward_output_norm,
            residual_multiplier: architecture.residual_multiplier,
            parallel: architecture.parallel,
            scales: architecture.scaled_residuals.map(|scales| BlockScales {
                from_normed: scales.from_normed,
                mixer: scales.full_attention,
                feed_forward: scales.feed_forward,
            }),
        })
    }
}

/// The feed-forward layer `layer` has: the routed experts on a layer the
/// mixture covers, the dense feed-forward otherwise.
fn feed_forward_block(
    builder: &VarBuilder<'_>,
    config: &Config,
    architecture: &Architecture,
    layer: usize,
    adapters: &Adapters,
) -> candle_core::Result<FeedForwardBlock> {
    Ok(match &architecture.experts {
        Some(experts) if architecture.routed(layer) => FeedForwardBlock::Routed(Experts::load(
            builder,
            config.hidden_size,
            experts,
            architecture.activation,
        )?),
        _ => FeedForwardBlock::Dense(FeedForward::load(builder, config, architecture, layer, adapters)?),
    })
}

impl DecoderLayer {
    /// The window this layer's attention looks through, if any.
    pub(super) fn window(&self) -> Option<usize> {
        match &self.mixer {
            Mixer::Attention(attention) => attention.window(),
            Mixer::Parallel(mixers) => mixers.attention.window(),
            Mixer::StateSpace(_)
            | Mixer::Structured(_)
            | Mixer::ShortConv(_)
            | Mixer::DeltaRule(_)
            | Mixer::Lightning(_)
            | Mixer::FeedForward(_) => None,
        }
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
        let mixed = match &self.mixer {
            Mixer::Attention(attention) => {
                attention.forward(&normed, index_pos, layer, cache, mask, mode)?
            }
            Mixer::StateSpace(state_space) => state_space.forward(&normed, layer, cache)?,
            Mixer::Structured(structured) => structured.forward(&normed, layer, cache)?,
            Mixer::ShortConv(convolution) => convolution.forward(&normed, layer, cache)?,
            Mixer::DeltaRule(delta) => delta.forward(&normed, layer, cache)?,
            Mixer::Lightning(lightning) => lightning.forward(&normed, layer, cache)?,
            Mixer::FeedForward(feed_forward) => feed_forward.forward(&normed, mode.route)?,
            Mixer::Parallel(mixers) => mixers.forward(&normed, index_pos, layer, cache, mask, mode)?,
        };
        let attention = optional_norm(self.attention_output_norm.as_ref(), &mixed, mode.pass)?;
        let Some(feed_forward_block) = &self.feed_forward else {
            return hidden + self.scaled(attention)?;
        };
        // MiniMax-Text-01: `residual · alpha + output · beta` after each
        // sublayer, the residual being the sublayer's normed input under
        // `postnorm`.
        if let Some(BlockScales { from_normed, mixer, feed_forward }) = self.scales {
            let join = |residual: &Tensor, output: Tensor, (alpha, beta): (f64, f64)| {
                (residual * alpha)? + (output * beta)?
            };
            let hidden = join(if from_normed { &normed } else { hidden }, attention, mixer)?;
            let normed = optional_norm(self.feed_forward_norm.as_ref(), &hidden, mode.pass)?;
            let output = feed_forward_block.forward(&normed, mode.route)?;
            return join(if from_normed { &normed } else { &hidden }, output, feed_forward);
        }
        if self.parallel {
            // GPT-NeoX normalises the feed-forward's input on its own; the
            // other parallel families reuse attention's.
            let feed_forward_input = match &self.feed_forward_norm {
                Some(norm) => norm.forward(hidden, mode.pass)?,
                None => normed,
            };
            let feed_forward = feed_forward_block.forward(&feed_forward_input, mode.route)?;
            return (hidden + self.scaled(attention)?)? + self.scaled(feed_forward)?;
        }
        let hidden = (hidden + self.scaled(attention)?)?;
        let feed_forward = feed_forward_block.forward(
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
