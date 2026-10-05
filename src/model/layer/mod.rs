//! One decoder block: the feed-forward half, the norms around both halves,
//! and how their outputs join the residual stream.

pub(super) mod experts;
pub(super) mod norm;
mod recurrent;
pub(super) mod shared;

use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::{
    Activation, Architecture, Cache, DeltaRuleForm, FeedForwardKind, Mode, ParallelScan, Pass, Route,
    attention::{Attention, project},
};
use experts::Experts;
use norm::{Norm, NormSpec};
use shared::{SharedBlock, SharedInvocation};
use recurrent::{
    delta::DeltaRule,
    lightning::Lightning,
    state_space::{ShortConv, StateSpace},
    structured::Structured,
};

/// The feed-forward half of a block: one dense feed-forward, a router over
/// experts, Gemma 4's dense feed-forward beside its experts, or a
/// LongCat-Flash half-layer's dense feed-forward with the stored layer's
/// shortcut experts.
#[derive(Debug, Clone)]
enum FeedForwardBlock {
    Dense(FeedForward),
    Routed(Experts),
    Paired(Box<PairedExperts>),
    /// The first half's experts read its feed-forward input and are kept in
    /// the cache; the second half (`None`) adds them to its output.
    Shortcut(FeedForward, Option<Experts>),
}

impl FeedForwardBlock {
    /// `hidden` is the normed input, except for a paired block, which norms
    /// the residual itself for each of its halves. A shortcut block runs
    /// its dense feed-forward here; [`FeedForwardBlock::shortcut`] runs the
    /// rest.
    fn forward(&self, hidden: &Tensor, mode: Mode) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(dense) | Self::Shortcut(dense, _) => dense.forward(hidden, mode.route),
            Self::Routed(experts) => experts.forward(hidden),
            Self::Paired(paired) => {
                let dense = paired.dense.forward(&paired.dense_norm.forward(hidden, mode.pass)?, mode.route)?;
                let dense = paired.dense_output_norm.forward(&dense, mode.pass)?;
                let routed = paired.experts.forward_routed(
                    &paired.router_norm.forward(hidden, mode.pass)?,
                    &paired.experts_norm.forward(hidden, mode.pass)?,
                )?;
                dense + paired.experts_output_norm.forward(&routed, mode.pass)?
            }
        }
    }

    /// LongCat-Flash's shortcut: the first half keeps its experts' output
    /// over `input` (its feed-forward input) in the cache and adds nothing;
    /// the second half takes it and adds it to `output`. Every other block
    /// returns `output` as it is.
    fn shortcut(&self, input: &Tensor, output: Tensor, layer: usize, cache: &mut Cache) -> candle_core::Result<Tensor> {
        match self {
            Self::Shortcut(_, Some(experts)) => {
                cache.shortcut = Some(experts.forward(input)?);
                Ok(output)
            }
            Self::Shortcut(_, None) => match cache.shortcut.take() {
                Some(kept) => output + kept,
                None => candle_core::bail!("layer {layer} closes a shortcut that no earlier layer opened"),
            },
            _ => Ok(output),
        }
    }
}

/// Gemma 4's experts beside its dense feed-forward: the dense half reads the
/// residual through `pre_feedforward_layernorm` and is normed by
/// `post_feedforward_layernorm_1`; the router reads it through a scale-free
/// norm, the experts through `pre_feedforward_layernorm_2`, and their sum
/// is normed by `post_feedforward_layernorm_2`.
#[derive(Debug, Clone)]
struct PairedExperts {
    dense: FeedForward,
    dense_norm: Norm,
    dense_output_norm: Norm,
    experts: Experts,
    router_norm: Norm,
    experts_norm: Norm,
    experts_output_norm: Norm,
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
    /// xIELU's parameters, when the activation is xIELU (Apertus).
    xielu: Option<Xielu>,
}

/// xIELU's parameters from the feed-forward's `act_fn`, as Transformers'
/// `XIELUActivation` uses them: `alpha_p` is the softplus of the stored
/// one, `alpha_n` is `beta` plus the softplus of the stored one.
#[derive(Debug, Clone, Copy)]
struct Xielu {
    alpha_p: f64,
    alpha_n: f64,
    beta: f64,
    eps: f64,
}

impl Xielu {
    fn load(builder: VarBuilder<'_>) -> candle_core::Result<Self> {
        let scalar = |name: &str| -> candle_core::Result<f64> {
            let values = builder.get_unchecked(name)?.flatten_all()?.to_dtype(candle_core::DType::F64)?.to_vec1::<f64>()?;
            match values.as_slice() {
                [value] => Ok(*value),
                _ => candle_core::bail!("xIELU's {name} holds {} values, not one", values.len()),
            }
        };
        let softplus = |value: f64| value.exp().ln_1p();
        let beta = scalar("beta")?;
        Ok(Self {
            alpha_p: softplus(scalar("alpha_p")?),
            alpha_n: beta + softplus(scalar("alpha_n")?),
            beta,
            eps: scalar("eps")?,
        })
    }

    /// `alpha_p · x² + beta · x` above zero, and
    /// `(expm1(min(x, eps)) − x) · alpha_n + beta · x` elsewhere.
    fn apply(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let linear = (input * self.beta)?;
        let positive = ((input.sqr()? * self.alpha_p)? + &linear)?;
        let negative = ((((input.minimum(self.eps)?.exp()? - 1.0)? - input)? * self.alpha_n)? + &linear)?;
        input.gt(0.0)?.where_cond(&positive, &negative)
    }
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
        // DeciLM states each layer's feed-forward width apart.
        let intermediate = architecture
            .layer_plan(layer)
            .and_then(|plan| plan.intermediate)
            .unwrap_or(config.intermediate_size);
        let hidden = config.hidden_size;
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
            xielu: if architecture.activation == Activation::Xielu {
                let parent = names.down.rsplit_once('.').map_or("", |(parent, _)| parent);
                let block = if parent.is_empty() { builder.clone() } else { builder.pp(parent) };
                Some(Xielu::load(block.pp("act_fn"))?)
            } else {
                None
            },
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
                (self.activate(&gate)? * up)?
            }
            None => self.activate(&up)?,
        };
        let output = project(&self.down, self.down_adapter.as_ref(), &inner, route)?;
        match self.scales {
            Some((_, output_scale)) => output * output_scale,
            None => Ok(output),
        }
    }

    fn activate(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        match &self.xielu {
            Some(xielu) => xielu.apply(input),
            None => self.activation.apply(input),
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
    /// Zamba2's hybrid layers: a shared block feeding a Mamba-2 scan.
    Hybrid(Box<Hybrid>),
    /// No mixer at all (DeciLM's no-op attention).
    Skip,
}

/// A Zamba2 hybrid layer's mixer: the shared block over the hidden state
/// beside the embeddings, projected back and added to the hidden state,
/// then the norm and the Mamba-2 scan.
#[derive(Debug, Clone)]
struct Hybrid {
    shared: SharedInvocation,
    norm: Norm,
    scan: Structured,
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

/// One decoder layer: its block, and what Gemma 4 adds after it — the
/// per-layer input and the `layer_scalar` the whole output is multiplied by.
#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    block: Block,
    per_layer_input: Option<PerLayerInput>,
    scalar: Option<Tensor>,
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

impl DecoderLayer {
    /// `shared` is, on a Zamba2 hybrid layer, the shared block it uses and
    /// that block's builder (`model.layers.{owner}.shared_transformer`).
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
        shared: Option<(&SharedBlock, &VarBuilder<'_>)>,
    ) -> candle_core::Result<Self> {
        let per_layer_input = match architecture.per_layer_input {
            Some(spec) => Some(PerLayerInput {
                gate: linear_no_bias(config.hidden_size, spec.width, builder.pp("per_layer_input_gate"))?,
                projection: linear_no_bias(spec.width, config.hidden_size, builder.pp("per_layer_projection"))?,
                norm: NormSpec::of(architecture)
                    .load(config.hidden_size, builder.pp("post_per_layer_input_norm"))?,
                activation: architecture.activation,
            }),
            None => None,
        };
        let scalar = if architecture.layer_scalar { Some(builder.get(1, "layer_scalar")?) } else { None };
        Ok(Self {
            block: Block::load(builder, config, architecture, layer, adapters, shared)?,
            per_layer_input,
            scalar,
        })
    }

    /// The window this layer's attention looks through, if any.
    pub(super) fn window(&self) -> Option<usize> {
        self.block.window()
    }

    pub(super) fn forward(
        &self,
        hidden: &Tensor,
        inputs: &LayerInputs<'_>,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let output = self.block.forward(hidden, inputs.embedded, index_pos, layer, cache, mask, mode)?;
        let output = match (&self.per_layer_input, &inputs.per_layer) {
            (Some(per_layer), Some(input)) => {
                let gated = (per_layer.activation.apply(&per_layer.gate.forward(&output)?)? * input)?;
                let added = per_layer.norm.forward(&per_layer.projection.forward(&gated)?, mode.pass)?;
                (output + added)?
            }
            (Some(_), None) => candle_core::bail!("layer {layer} takes a per-layer input and was given none"),
            (None, _) => output,
        };
        match &self.scalar {
            Some(scalar) => output.broadcast_mul(scalar),
            None => Ok(output),
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
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

impl Block {
    fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
        shared: Option<(&SharedBlock, &VarBuilder<'_>)>,
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
                    spec,
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
                        DeltaRuleForm::Qwen3Next | DeltaRuleForm::Qwen35 | DeltaRuleForm::OlmoHybrid => "linear_attn",
                        DeltaRuleForm::Kimi => "self_attn",
                        DeltaRuleForm::Ling => "attention",
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
        // Zamba2's hybrid layers: the shared block's output, projected by the
        // layer's `linear`, joins the hidden state before the Mamba-2 mixer's
        // norm (`mamba_decoder.input_layernorm`), and the mixer's output is
        // added to the hidden state.
        if let (Some((block, block_builder)), Some(state_space)) = (shared, architecture.state_space.as_ref()) {
            let Some(heads) = state_space.structured else {
                candle_core::bail!("a Zamba2 hybrid layer runs Mamba-2 heads, and this architecture states none");
            };
            let Some(blocks) = architecture.shared_blocks else {
                candle_core::bail!("a hybrid layer needs the architecture's shared blocks");
            };
            let decoder = builder.pp("mamba_decoder");
            return Ok(Self {
                attention_norm: None,
                mixer: Mixer::Hybrid(Box::new(Hybrid {
                    shared: SharedInvocation::load(
                        block,
                        block_builder,
                        &builder,
                        config.hidden_size,
                        blocks.slot(layer),
                    )?,
                    norm: spec.load(config.hidden_size, decoder.pp(names.attention_norm))?,
                    scan: Structured::load(
                        decoder.pp(names.state_space),
                        config.hidden_size,
                        config.rms_norm_eps,
                        state_space,
                        heads,
                    )?,
                })),
                attention_output_norm: None,
                feed_forward_norm: None,
                feed_forward: None,
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
        // DeciLM (Nemotron Super and Ultra): each layer states its own
        // key-value heads and feed-forward width, and either half may be
        // absent — no norm, no weights, the residual passed through.
        if let Some(plan) = architecture.layer_plan(layer) {
            let (attention_norm, mixer) = match plan.key_value_heads {
                Some(_) => (
                    Some(norm(names.attention_norm)?),
                    Mixer::Attention(Attention::load(&builder, config, architecture, layer, adapters)?),
                ),
                None => (None, Mixer::Skip),
            };
            let (feed_forward_norm, feed_forward) = match plan.intermediate {
                Some(_) => (
                    Some(norm(names.feed_forward_norm)?),
                    Some(FeedForwardBlock::Dense(FeedForward::load(&builder, config, architecture, layer, adapters)?)),
                ),
                None => (None, None),
            };
            return Ok(Self {
                attention_norm,
                mixer,
                attention_output_norm: None,
                feed_forward_norm,
                feed_forward,
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
                (false, true, true) => {
                    // MuseGlimmer's output norms keep their own epsilon
                    // (`post_norm_eps`).
                    let output_spec = NormSpec { eps: architecture.output_norm_eps.unwrap_or(spec.eps), ..spec };
                    let output_norm = |name: &str| output_spec.load(config.hidden_size, builder.pp(name));
                    (
                        Some(norm(names.attention_norm)?),
                        Some(output_norm(names.attention_output_norm)?),
                        Some(norm(names.feed_forward_norm)?),
                        Some(output_norm(names.feed_forward_output_norm)?),
                    )
                }
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
            // Gemma 4's paired block norms the residual itself.
            feed_forward_norm: if architecture.side_experts.is_some() { None } else { feed_forward_norm },
            feed_forward: Some(feed_forward_block(&builder, config, architecture, layer, adapters)?),
            feed_forward_output_norm: match feed_forward_output_norm {
                Some(existing) => Some(existing),
                None if architecture.routed_output_norm
                    && architecture.experts.is_some()
                    && architecture.routed(layer) =>
                {
                    Some(norm(names.feed_forward_output_norm)?)
                }
                None => None,
            },
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
    if let Some(experts) = &architecture.side_experts {
        let spec = NormSpec::of(architecture);
        let norm = |name: &str| spec.load(config.hidden_size, builder.pp(name));
        return Ok(FeedForwardBlock::Paired(Box::new(PairedExperts {
            dense: FeedForward::load(builder, config, architecture, layer, adapters)?,
            dense_norm: norm(architecture.names.feed_forward_norm)?,
            dense_output_norm: norm("post_feedforward_layernorm_1")?,
            experts: Experts::load(builder, config.hidden_size, experts, architecture.activation, layer)?,
            router_norm: spec.unscaled(config.hidden_size, builder)?,
            experts_norm: norm("pre_feedforward_layernorm_2")?,
            experts_output_norm: norm("post_feedforward_layernorm_2")?,
        })));
    }
    // LongCat-Flash: each half-layer's dense `mlps.{half}`, and on the first
    // half the stored layer's shortcut experts.
    if let Some(experts) = &architecture.shortcut_experts {
        let dense = FeedForward::load(builder, config, architecture, layer, adapters)?;
        let shortcut = if layer % 2 == 0 {
            Some(Experts::load(builder, config.hidden_size, experts, architecture.activation, layer)?)
        } else {
            None
        };
        return Ok(FeedForwardBlock::Shortcut(dense, shortcut));
    }
    Ok(match architecture.experts_at(layer) {
        Some(experts) if architecture.routed(layer) => FeedForwardBlock::Routed(Experts::load(
            builder,
            config.hidden_size,
            &experts,
            architecture.activation,
            layer,
        )?),
        _ => FeedForwardBlock::Dense(FeedForward::load(builder, config, architecture, layer, adapters)?),
    })
}

impl Block {
    /// The window this block's attention looks through, if any.
    fn window(&self) -> Option<usize> {
        match &self.mixer {
            Mixer::Attention(attention) => attention.window(),
            Mixer::Parallel(mixers) => mixers.attention.window(),
            Mixer::Hybrid(_) | Mixer::Skip => None,
            Mixer::StateSpace(_)
            | Mixer::Structured(_)
            | Mixer::ShortConv(_)
            | Mixer::DeltaRule(_)
            | Mixer::Lightning(_)
            | Mixer::FeedForward(_) => None,
        }
    }

    /// `embedded` is the token embeddings before the first layer, which
    /// Zamba2's hybrid layers read beside the hidden state.
    pub(super) fn forward(
        &self,
        hidden: &Tensor,
        embedded: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        // DeciLM's no-op attention: the block is its feed-forward alone, or
        // nothing at all.
        if let Mixer::Skip = self.mixer {
            let Some(feed_forward_block) = &self.feed_forward else {
                return Ok(hidden.clone());
            };
            let normed = optional_norm(self.feed_forward_norm.as_ref(), hidden, mode.pass)?;
            return hidden + self.scaled(feed_forward_block.forward(&normed, mode)?)?;
        }
        let normed = optional_norm(self.attention_norm.as_ref(), hidden, mode.pass)?;
        let mixed = match &self.mixer {
            Mixer::Skip => candle_core::bail!("layer {layer} has no attention to run"),
            Mixer::Attention(attention) => {
                attention.forward(&normed, index_pos, layer, cache, mask, mode)?
            }
            Mixer::StateSpace(state_space) => state_space.forward(&normed, layer, cache)?,
            Mixer::Structured(structured) => structured.forward(&normed, layer, cache)?,
            Mixer::ShortConv(convolution) => convolution.forward(&normed, layer, cache)?,
            Mixer::DeltaRule(delta) => delta.forward(&normed, layer, cache)?,
            Mixer::Lightning(lightning) => lightning.forward(&normed, index_pos, layer, cache, mode)?,
            Mixer::FeedForward(feed_forward) => feed_forward.forward(&normed, mode)?,
            Mixer::Parallel(mixers) => mixers.forward(&normed, index_pos, layer, cache, mask, mode)?,
            Mixer::Hybrid(hybrid) => {
                let shared = hybrid.shared.forward(&normed, embedded, index_pos, layer, cache, mask, mode)?;
                let joined = hybrid.norm.forward(&(&normed + shared)?, mode.pass)?;
                hybrid.scan.forward(&joined, layer, cache)?
            }
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
            let output = feed_forward_block.forward(&normed, mode)?;
            return join(if from_normed { &normed } else { &hidden }, output, feed_forward);
        }
        if self.parallel {
            // GPT-NeoX normalises the feed-forward's input on its own; the
            // other parallel families reuse attention's.
            let feed_forward_input = match &self.feed_forward_norm {
                Some(norm) => norm.forward(hidden, mode.pass)?,
                None => normed,
            };
            let feed_forward = feed_forward_block.forward(&feed_forward_input, mode)?;
            return (hidden + self.scaled(attention)?)? + self.scaled(feed_forward)?;
        }
        let hidden = (hidden + self.scaled(attention)?)?;
        let feed_forward_input = optional_norm(self.feed_forward_norm.as_ref(), &hidden, mode.pass)?;
        let feed_forward = feed_forward_block.forward(&feed_forward_input, mode)?;
        let feed_forward =
            optional_norm(self.feed_forward_output_norm.as_ref(), &feed_forward, mode.pass)?;
        let output = (hidden + self.scaled(feed_forward)?)?;
        feed_forward_block.shortcut(&feed_forward_input, output, layer, cache)
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
