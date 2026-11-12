//! What mixes information across positions in a block: attention, a scan,
//! a convolution, a linear attention, or two of them side by side.

use candle_core::Tensor;

use super::super::{
    Cache, Mode, ParallelScan,
    attention::{Attention, relative::RelativeAttention},
    deepseek4::attention::CompressedAttention,
};
use super::feed_forward::FeedForwardBlock;
use super::norm::Norm;
use super::recurrent::{
    delta::DeltaRule, lightning::Lightning, short_conv::ShortConv, state_space::StateSpace,
    structured::Structured,
};
use super::shared::SharedInvocation;

/// What mixes information across positions in a block.
#[derive(Debug, Clone)]
pub(super) enum Mixer {
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
    /// Inkling's relative-position attention.
    Relative(Box<RelativeAttention>),
    /// DeepSeek-V4's compressed attention.
    Compressed(Box<CompressedAttention>),
}

/// A Zamba2 hybrid layer's mixer: the shared block over the hidden state
/// beside the embeddings, projected back and added to the hidden state,
/// then the norm and the Mamba-2 scan.
#[derive(Debug, Clone)]
pub(super) struct Hybrid {
    pub(super) shared: SharedInvocation,
    pub(super) norm: Norm,
    pub(super) scan: Structured,
}

/// Falcon-H1's two mixers on one normed input, each output scaled before
/// the two are added.
#[derive(Debug, Clone)]
pub(super) struct ParallelMixers {
    pub(super) attention: Attention,
    pub(super) scan: Structured,
    pub(super) scales: ParallelScan,
}

impl ParallelMixers {
    pub(super) fn forward(
        &self,
        normed: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let ParallelScan {
            attention_in,
            attention_out,
            scan_out,
        } = self.scales;
        let scanned = (self.scan.forward(normed, layer, cache)? * scan_out)?;
        let attended = self.attention.forward(
            &(normed * attention_in)?,
            index_pos,
            layer,
            cache,
            mask,
            mode,
        )?;
        scanned + (attended * attention_out)?
    }
}

