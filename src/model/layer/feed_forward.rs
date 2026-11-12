//! The feed-forward half of a block: one dense feed-forward, a router over
//! experts, Gemma 4's dense feed-forward beside its experts, LongCat-Flash's
//! half-layer with the stored layer's shortcut experts, or Inkling's.

use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::Config;

use crate::failures::{model::decoder::SHORTCUT_UNOPENED, refuse_candle};
use crate::lora::Adapters;

use super::super::{Architecture, Cache, Mode, inkling};
use super::dense::FeedForward;
use super::experts::Experts;
use super::norm::{Norm, NormSpec};

/// The feed-forward half of a block.
#[derive(Debug, Clone)]
pub(super) enum FeedForwardBlock {
    Dense(FeedForward),
    Routed(Experts),
    Paired(Box<PairedExperts>),
    /// The first half's experts read its feed-forward input and are kept in
    /// the cache; the second half (`None`) adds them to its output.
    Shortcut(FeedForward, Option<Experts>),
    /// Inkling's dense or routed feed-forward and its residual convolution.
    Inkling(Box<inkling::FeedForward>),
}

/// Gemma 4's experts beside its dense feed-forward: the dense half reads the
/// residual through `pre_feedforward_layernorm` and is normed by
/// `post_feedforward_layernorm_1`; the router reads it through a scale-free
/// norm, the experts through `pre_feedforward_layernorm_2`, and their sum
/// is normed by `post_feedforward_layernorm_2`.
#[derive(Debug, Clone)]
pub(super) struct PairedExperts {
    dense: FeedForward,
    dense_norm: Norm,
    dense_output_norm: Norm,
    experts: Experts,
    router_norm: Norm,
    experts_norm: Norm,
    experts_output_norm: Norm,
}

impl FeedForwardBlock {
    /// The feed-forward of layer `layer` over `hidden`, the normed input,
    /// except for a paired block, which norms the residual itself for each
    /// of its halves. An Inkling feed-forward keeps its convolution history
    /// in the cache, and routing by token id reads the call's ids there. A
    /// shortcut block runs its dense feed-forward here;
    /// [`FeedForwardBlock::shortcut`] runs the rest.
    pub(super) fn forward_at(
        &self,
        hidden: &Tensor,
        mode: Mode,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Inkling(feed_forward) => feed_forward.forward(hidden, layer, cache),
            Self::Routed(experts) => experts.forward_ids(hidden, cache.token_ids.as_deref()),
            Self::Dense(dense) | Self::Shortcut(dense, _) => dense.forward(hidden, mode.route),
            Self::Paired(paired) => {
                let dense = paired
                    .dense
                    .forward(&paired.dense_norm.forward(hidden, mode.pass)?, mode.route)?;
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
    pub(super) fn shortcut(
        &self,
        input: &Tensor,
        output: Tensor,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Shortcut(_, Some(experts)) => {
                cache.shortcut = Some(experts.forward(input)?);
                Ok(output)
            }
            Self::Shortcut(_, None) => match cache.shortcut.take() {
                Some(kept) => output + kept,
                None => Err(refuse_candle(
                    &SHORTCUT_UNOPENED,
                    &[
                        ("layer", &layer.to_string()),
                        ("first", &layer.saturating_sub(1).to_string()),
                    ],
                )),
            },
            _ => Ok(output),
        }
    }
}

/// The feed-forward layer `layer` has: the routed experts on a layer the
/// mixture covers, the dense feed-forward otherwise.
pub(super) fn feed_forward_block(
    builder: &VarBuilder<'_>,
    config: &Config,
    architecture: &Architecture,
    layer: usize,
    adapters: &Adapters,
) -> candle_core::Result<FeedForwardBlock> {
    if let Some(spec) = &architecture.inkling {
        return Ok(FeedForwardBlock::Inkling(Box::new(
            inkling::FeedForward::load(
                builder,
                config.hidden_size,
                config.intermediate_size,
                spec,
                architecture.activation,
                layer,
            )?,
        )));
    }
    if let Some(experts) = &architecture.side_experts {
        let spec = NormSpec::of(architecture);
        let norm = |name: &str| spec.load(config.hidden_size, builder.pp(name));
        return Ok(FeedForwardBlock::Paired(Box::new(PairedExperts {
            dense: FeedForward::load(builder, config, architecture, layer, adapters)?,
            dense_norm: norm(architecture.names.feed_forward_norm)?,
            dense_output_norm: norm("post_feedforward_layernorm_1")?,
            experts: Experts::load(
                builder,
                config.hidden_size,
                experts,
                architecture.activation,
                layer,
            )?,
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
            Some(Experts::load(
                builder,
                config.hidden_size,
                experts,
                architecture.activation,
                layer,
            )?)
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
        _ => FeedForwardBlock::Dense(FeedForward::load(
            builder,
            config,
            architecture,
            layer,
            adapters,
        )?),
    })
}
