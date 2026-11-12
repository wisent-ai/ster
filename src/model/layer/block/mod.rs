//! One block of a decoder layer: the mixer and the feed-forward, the norms
//! around both halves, and how their outputs join the residual stream.

mod load;
mod norms;

use candle_core::Tensor;

use super::super::{Cache, Mode, Pass};
use super::altup;
use super::feed_forward::FeedForwardBlock;
use super::mixer::Mixer;
use super::norm::Norm;

#[derive(Debug, Clone)]
pub(super) struct Block {
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
    /// Gemma 3n's Laurel beside attention.
    laurel: Option<altup::Laurel>,
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
    /// The window this block's attention looks through, if any.
    pub(super) fn window(&self) -> Option<usize> {
        match &self.mixer {
            Mixer::Attention(attention) => attention.window(),
            Mixer::Parallel(mixers) => mixers.attention.window(),
            Mixer::Hybrid(_) | Mixer::Skip => None,
            Mixer::Relative(attention) => attention.window(),
            Mixer::Compressed(attention) => attention.window(),
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
            return hidden
                + self.scaled(feed_forward_block.forward_at(&normed, mode, layer, cache)?)?;
        }
        let normed = optional_norm(self.attention_norm.as_ref(), hidden, mode.pass)?;
        let mixed = self.mix(&normed, embedded, index_pos, layer, cache, mask, mode)?;
        let attention = optional_norm(self.attention_output_norm.as_ref(), &mixed, mode.pass)?;
        let Some(feed_forward_block) = &self.feed_forward else {
            return hidden + self.scaled(attention)?;
        };
        // MiniMax-Text-01: `residual · alpha + output · beta` after each
        // sublayer, the residual being the sublayer's normed input under
        // `postnorm`.
        if let Some(BlockScales {
            from_normed,
            mixer,
            feed_forward,
        }) = self.scales
        {
            let join = |residual: &Tensor, output: Tensor, (alpha, beta): (f64, f64)| {
                (residual * alpha)? + (output * beta)?
            };
            let hidden = join(if from_normed { &normed } else { hidden }, attention, mixer)?;
            let normed = optional_norm(self.feed_forward_norm.as_ref(), &hidden, mode.pass)?;
            let output = feed_forward_block.forward_at(&normed, mode, layer, cache)?;
            return join(
                if from_normed { &normed } else { &hidden },
                output,
                feed_forward,
            );
        }
        if self.parallel {
            // GPT-NeoX normalises the feed-forward's input on its own; the
            // other parallel families reuse attention's.
            let feed_forward_input = match &self.feed_forward_norm {
                Some(norm) => norm.forward(hidden, mode.pass)?,
                None => normed,
            };
            let feed_forward =
                feed_forward_block.forward_at(&feed_forward_input, mode, layer, cache)?;
            return (hidden + self.scaled(attention)?)? + self.scaled(feed_forward)?;
        }
        let hidden = (hidden + self.scaled(attention)?)?;
        // Gemma 3n averages attention's residual with Laurel's.
        let hidden = match &self.laurel {
            Some(laurel) => {
                ((hidden + laurel.forward(&normed, mode.pass)?)? / std::f64::consts::SQRT_2)?
            }
            None => hidden,
        };
        let feed_forward_input =
            optional_norm(self.feed_forward_norm.as_ref(), &hidden, mode.pass)?;
        let feed_forward =
            feed_forward_block.forward_at(&feed_forward_input, mode, layer, cache)?;
        let feed_forward = optional_norm(
            self.feed_forward_output_norm.as_ref(),
            &feed_forward,
            mode.pass,
        )?;
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

    /// The block's mixer over its normed input.
    fn mix(
        &self,
        normed: &Tensor,
        embedded: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        Ok(match &self.mixer {
            // DeciLM's no-op attention adds nothing to the residual.
            Mixer::Skip => normed.zeros_like()?,
            Mixer::Attention(attention) => {
                attention.forward(normed, index_pos, layer, cache, mask, mode)?
            }
            Mixer::StateSpace(state_space) => state_space.forward(normed, layer, cache)?,
            Mixer::Structured(structured) => structured.forward(normed, layer, cache)?,
            Mixer::ShortConv(convolution) => convolution.forward(normed, layer, cache)?,
            Mixer::DeltaRule(delta) => delta.forward(normed, layer, cache)?,
            Mixer::Lightning(lightning) => {
                lightning.forward(normed, index_pos, layer, cache, mode)?
            }
            Mixer::FeedForward(feed_forward) => {
                feed_forward.forward_at(normed, mode, layer, cache)?
            }
            Mixer::Parallel(mixers) => {
                mixers.forward(normed, index_pos, layer, cache, mask, mode)?
            }
            Mixer::Hybrid(hybrid) => {
                let shared = hybrid
                    .shared
                    .forward(normed, embedded, index_pos, layer, cache, mask, mode)?;
                let joined = hybrid.norm.forward(&(normed + shared)?, mode.pass)?;
                hybrid.scan.forward(&joined, layer, cache)?
            }
            Mixer::Relative(attention) => {
                attention.forward(normed, index_pos, layer, cache, mask, mode)?
            }
            Mixer::Compressed(attention) => {
                attention.forward(normed, index_pos, layer, cache, mask, mode)?
            }
        })
    }

    /// The first sublayer's output alone, before any residual: the norm,
    /// the mixer and the output norm.
    pub(super) fn mixer_output(
        &self,
        input: &Tensor,
        embedded: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let normed = optional_norm(self.attention_norm.as_ref(), input, mode.pass)?;
        let mixed = self.mix(&normed, embedded, index_pos, layer, cache, mask, mode)?;
        optional_norm(self.attention_output_norm.as_ref(), &mixed, mode.pass)
    }

    /// The feed-forward's output alone, before any residual: nothing, on a
    /// block whose feed-forward is absent (a state-space block, DeciLM's
    /// no-op feed-forward).
    pub(super) fn feed_forward_output(
        &self,
        input: &Tensor,
        layer: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let Some(feed_forward_block) = &self.feed_forward else {
            return input.zeros_like();
        };
        let normed = optional_norm(self.feed_forward_norm.as_ref(), input, mode.pass)?;
        let output = feed_forward_block.forward_at(&normed, mode, layer, cache)?;
        optional_norm(self.feed_forward_output_norm.as_ref(), &output, mode.pass)
    }
}

/// `hidden` through `norm` when the block has one at this position.
fn optional_norm(norm: Option<&Norm>, hidden: &Tensor, pass: Pass) -> candle_core::Result<Tensor> {
    match norm {
        Some(norm) => norm.forward(hidden, pass),
        None => Ok(hidden.clone()),
    }
}
