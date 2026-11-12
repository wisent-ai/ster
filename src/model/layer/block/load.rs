//! Building one block from the config's reading and the weights: the mixer
//! its family runs on this layer, the feed-forward and the norms.

use candle_nn::VarBuilder;
use candle_transformers::models::llama::Config;

use crate::lora::Adapters;

use super::super::super::{Architecture, DeltaRuleForm, attention::Attention};
use super::super::dense::FeedForward;
use super::super::feed_forward::{FeedForwardBlock, feed_forward_block};
use super::super::mixer::{Hybrid, Mixer, ParallelMixers};
use super::super::norm::{Norm, NormSpec};
use super::super::recurrent::{
    delta::DeltaRule, lightning::Lightning, short_conv::ShortConv, state_space::StateSpace,
    structured::Structured,
};
use super::super::shared::{SharedBlock, SharedInvocation};
use super::{Block, BlockScales};

/// What every part of one block is built from.
pub(super) struct Loader<'a> {
    pub(super) builder: VarBuilder<'a>,
    pub(super) config: &'a Config,
    pub(super) architecture: &'a Architecture,
    pub(super) layer: usize,
    pub(super) adapters: &'a Adapters,
    pub(super) spec: NormSpec,
}

impl Loader<'_> {
    /// The block norm the family names `name`.
    pub(super) fn norm(&self, name: &str) -> candle_core::Result<Norm> {
        self.spec.load(self.config.hidden_size, self.builder.pp(name))
    }

    pub(super) fn feed_forward(&self) -> candle_core::Result<FeedForwardBlock> {
        feed_forward_block(
            &self.builder,
            self.config,
            self.architecture,
            self.layer,
            self.adapters,
        )
    }

    pub(super) fn attention(&self) -> candle_core::Result<Attention> {
        Attention::load(
            &self.builder,
            self.config,
            self.architecture,
            self.layer,
            self.adapters,
        )
    }

    /// A block of `mixer` behind `attention_norm` and an optional
    /// feed-forward behind `feed_forward_norm`, each half added to the
    /// residual as it is.
    fn plain(
        &self,
        attention_norm: Option<Norm>,
        mixer: Mixer,
        feed_forward_norm: Option<Norm>,
        feed_forward: Option<FeedForwardBlock>,
    ) -> Block {
        Block {
            attention_norm,
            mixer,
            attention_output_norm: None,
            feed_forward_norm,
            feed_forward,
            feed_forward_output_norm: None,
            residual_multiplier: self.architecture.residual_multiplier,
            parallel: false,
            scales: None,
            laurel: None,
        }
    }

    /// A normed `mixer` and a normed feed-forward, as MiniMax-Text-01's
    /// lightning layers, Qwen3-Next's linear-attention layers, LFM2's
    /// convolution layers and Falcon-H1's parallel layers run them.
    fn beside_feed_forward(&self, mixer: Mixer) -> candle_core::Result<Block> {
        let names = self.architecture.names;
        Ok(self.plain(
            Some(self.norm(names.attention_norm)?),
            mixer,
            Some(self.norm(names.feed_forward_norm)?),
            Some(self.feed_forward()?),
        ))
    }
}

impl Block {
    /// `shared` is, on a Zamba2 hybrid layer, the shared block it uses and
    /// that block's builder.
    pub(in crate::model::layer) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
        shared: Option<(&SharedBlock, &VarBuilder<'_>)>,
    ) -> candle_core::Result<Self> {
        let loader = Loader {
            builder,
            config,
            architecture,
            layer,
            adapters,
            spec: NormSpec::of(architecture),
        };
        match loader.recurrent(shared)? {
            Some(block) => Ok(block),
            None => loader.planned(),
        }
    }
}

impl Loader<'_> {
    /// A block whose mixer is not attention alone, or `None` on a layer
    /// that runs attention.
    fn recurrent(
        &self,
        shared: Option<(&SharedBlock, &VarBuilder<'_>)>,
    ) -> candle_core::Result<Option<Block>> {
        let (config, architecture, layer) = (self.config, self.architecture, self.layer);
        let (builder, names) = (&self.builder, architecture.names);
        let hidden = config.hidden_size;
        // MiniMax-Text-01's lightning-attention layers join the residual by
        // the layer type's scales.
        if let Some(&lightning) = architecture.lightning_at(layer) {
            let mixer = Mixer::Lightning(Lightning::load(
                builder.pp(names.attention),
                hidden,
                layer,
                config.num_hidden_layers,
                lightning,
                self.spec,
            )?);
            return Ok(Some(Block {
                scales: architecture.scaled_residuals.map(|scales| BlockScales {
                    from_normed: scales.from_normed,
                    mixer: scales.linear_attention,
                    feed_forward: scales.feed_forward,
                }),
                ..self.beside_feed_forward(mixer)?
            }));
        }
        if let Some(&spec) = architecture.delta_rule_at(layer) {
            let block = builder.pp(match spec.form {
                DeltaRuleForm::Qwen3Next | DeltaRuleForm::Qwen35 | DeltaRuleForm::OlmoHybrid => {
                    "linear_attn"
                }
                DeltaRuleForm::Kimi => "self_attn",
                DeltaRuleForm::Ling => "attention",
            });
            let mixer = DeltaRule::load(block, hidden, config.rms_norm_eps, spec)?;
            return self.beside_feed_forward(Mixer::DeltaRule(mixer)).map(Some);
        }
        if let Some(convolution) = architecture.short_convolution_at(layer) {
            let mixer = ShortConv::load(builder.pp(names.state_space), hidden, convolution)?;
            return self.beside_feed_forward(Mixer::ShortConv(mixer)).map(Some);
        }
        // The readings that run Zamba2's hybrid layers and Falcon-H1's
        // parallel scan state Mamba-2 heads and, for Zamba2, its shared
        // blocks; a layer runs them only where all are there.
        let structured = architecture
            .state_space
            .as_ref()
            .and_then(|state_space| state_space.structured.map(|heads| (state_space, heads)));
        // Zamba2: the shared block's output, projected by the layer's
        // `linear`, joins the hidden state before the Mamba-2 mixer's norm
        // (`mamba_decoder.input_layernorm`), and the mixer's output is added
        // to the hidden state.
        if let (Some((block, block_builder)), Some((state_space, heads)), Some(blocks)) =
            (shared, structured, architecture.shared_blocks)
        {
            let decoder = builder.pp("mamba_decoder");
            let hybrid = Hybrid {
                shared: SharedInvocation::load(
                    block,
                    block_builder,
                    builder,
                    hidden,
                    blocks.slot(layer),
                )?,
                norm: self.spec.load(hidden, decoder.pp(names.attention_norm))?,
                scan: Structured::load(
                    decoder.pp(names.state_space),
                    hidden,
                    config.rms_norm_eps,
                    state_space,
                    heads,
                )?,
            };
            return Ok(Some(self.plain(None, Mixer::Hybrid(Box::new(hybrid)), None, None)));
        }
        // A state-space block is one norm, the mixer and the residual add —
        // the whole block in Mamba; in Jamba a norm and a feed-forward follow
        // as in any other block.
        if let Some(state_space) = architecture.state_space_at(layer) {
            let scan = builder.pp(names.state_space);
            let eps = config.rms_norm_eps;
            let mixer = match state_space.structured {
                Some(heads) => {
                    Mixer::Structured(Structured::load(scan, hidden, eps, state_space, heads)?)
                }
                None => Mixer::StateSpace(StateSpace::load(scan, hidden, eps, state_space)?),
            };
            let (feed_forward_norm, feed_forward) = if state_space.feed_forward {
                (Some(self.norm(names.feed_forward_norm)?), Some(self.feed_forward()?))
            } else {
                (None, None)
            };
            let attention_norm = Some(self.norm(names.attention_norm)?);
            return Ok(Some(self.plain(attention_norm, mixer, feed_forward_norm, feed_forward)));
        }
        // Falcon-H1: attention and a Mamba-2 scan side by side on one normed
        // input, added to the residual, then a norm and the feed-forward.
        if let (Some(scales), Some((state_space, heads))) = (architecture.parallel_scan, structured)
        {
            let mixers = ParallelMixers {
                attention: self.attention()?,
                scan: Structured::load(
                    builder.pp(names.state_space),
                    hidden,
                    config.rms_norm_eps,
                    state_space,
                    heads,
                )?,
                scales,
            };
            return self.beside_feed_forward(Mixer::Parallel(Box::new(mixers))).map(Some);
        }
        Ok(None)
    }

    /// An attention block: Nemotron-H's lone sublayers, DeciLM's per-layer
    /// plan, or the family's attention and feed-forward between its norms.
    fn planned(&self) -> candle_core::Result<Block> {
        let (architecture, layer) = (self.architecture, self.layer);
        let names = architecture.names;
        // Nemotron-H: one norm and one sublayer per block — the feed-forward
        // alone on the layers that name it, attention alone on the others.
        if let Some(feed_forward_layers) = architecture.lone_sublayers {
            let mixer =
                if layer < u128::BITS as usize && feed_forward_layers & (1u128 << layer) != 0 {
                    Mixer::FeedForward(self.feed_forward()?)
                } else {
                    Mixer::Attention(self.attention()?)
                };
            return Ok(self.plain(Some(self.norm(names.attention_norm)?), mixer, None, None));
        }
        // DeciLM (Nemotron Super and Ultra): each layer states its own
        // key-value heads and feed-forward width, and either half may be
        // absent — no norm, no weights, the residual passed through.
        if let Some(plan) = architecture.layer_plan(layer) {
            let (attention_norm, mixer) = match plan.key_value_heads {
                Some(_) => (
                    Some(self.norm(names.attention_norm)?),
                    Mixer::Attention(self.attention()?),
                ),
                None => (None, Mixer::Skip),
            };
            let (feed_forward_norm, feed_forward) = match plan.intermediate {
                Some(_) => {
                    let dense = FeedForward::load(
                        &self.builder,
                        self.config,
                        architecture,
                        layer,
                        self.adapters,
                    )?;
                    (
                        Some(self.norm(names.feed_forward_norm)?),
                        Some(FeedForwardBlock::Dense(dense)),
                    )
                }
                None => (None, None),
            };
            return Ok(self.plain(attention_norm, mixer, feed_forward_norm, feed_forward));
        }
        self.standard()
    }
}
