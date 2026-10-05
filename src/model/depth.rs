//! Kimi-K3's attention residuals (`attn_res_block_size`): in place of adding
//! every sublayer's output to one running sum, the decoder keeps the sum of
//! each finished block of layers apart, and every sublayer reads a softmax
//! mix of those block sums and the current block's running sum, weighted by
//! a learned score per depth (`_apply_attn_res` in `modeling_kimi_linear.py`).
//!
//! A mix's score for one stored sum `v` is `rms(v) · (norm.weight ·
//! proj.weight)`: the sum RMS-normed without a scale, dotted with the norm's
//! scale times the one-row projection. Both are folded into one vector at
//! load. The mix is per position, so a decode step mixes only its own.

use candle_core::{D, DType, Tensor};
use candle_nn::VarBuilder;

/// One mix's folded score weights, `[hidden]` in F32.
#[derive(Debug, Clone)]
pub(super) struct DepthMix {
    weight: Tensor,
    eps: f64,
}

impl DepthMix {
    /// `norm` and `projection` are the mix's RMS norm (`…_res_norm`) and its
    /// one-row projection (`…_res_proj`), below `builder`.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        norm: &str,
        projection: &str,
        eps: f64,
    ) -> candle_core::Result<Self> {
        let scale = builder.pp(norm).get(hidden, "weight")?.to_dtype(DType::F32)?;
        let row = builder.pp(projection).get((1, hidden), "weight")?.to_dtype(DType::F32)?.squeeze(0)?;
        Ok(Self { weight: (scale * row)?, eps })
    }

    /// The softmax mix of `blocks` and `partial`, each `[batch, sequence,
    /// hidden]`, at `partial`'s dtype.
    pub(super) fn mix(&self, blocks: &[Tensor], partial: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = partial.dtype();
        let mut stored: Vec<&Tensor> = blocks.iter().collect();
        stored.push(partial);
        let values = Tensor::stack(&stored, 2)?.to_dtype(DType::F32)?;
        let variance = values.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = values.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let scores = normed.broadcast_mul(&self.weight)?.sum(D::Minus1)?;
        // softmax over the stored sums, composed so it has a backward pass.
        let probabilities = candle_nn::ops::softmax(&scores, D::Minus1)?.unsqueeze(D::Minus1)?;
        values.broadcast_mul(&probabilities)?.sum(2)?.to_dtype(dtype)
    }
}

/// A layer's two mixes, before attention and before the feed-forward, and
/// the block length: a layer whose index the length divides opens a new
/// block, storing the running sum it received.
#[derive(Debug, Clone)]
pub(super) struct DepthMixes {
    pub attention: DepthMix,
    pub feed_forward: DepthMix,
    pub block_size: usize,
}
