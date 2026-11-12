//! The scores of a query over its keys, the positional bias and masks laid
//! on them, and the softmax-weighted values they read.

use candle_core::{DType, Device, Tensor};

use super::super::{Cache, Mode, Pass, Readout};
use super::Attention;

impl Attention {
    /// The attention of `query` `[batch, heads, sequence, head_dim]` over
    /// `key` and `value` (key-value heads, not yet repeated), through
    /// `window` when it is a sliding-window attention, `[batch, heads,
    /// sequence, value_dim]` at the query's dtype.
    pub(super) fn attend(
        &self,
        query: &Tensor,
        key: Tensor,
        value: Tensor,
        window: Option<usize>,
        mask: Option<&Tensor>,
        index_pos: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (_, _, sequence, _) = query.dims4()?;
        let repeats = self.heads / self.key_value_heads;
        let key = repeat_key_value(key, repeats)?;
        let value = repeat_key_value(value, repeats)?;
        let input_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;
        let attention = (query.matmul(&key.t()?)? / self.score_divisor)?;
        // ALiBi adds `slope * key_position` to every score; within one query's
        // row that is the published `-slope * distance` plus a constant the
        // softmax ignores, and it is what BLOOM computes.
        let attention = match &self.alibi {
            // Step1 subtracts `slope · sqrt(distance)`, which is not a
            // constant shift within a row, so the distance is measured from
            // each query's absolute position; the last key sits at the last
            // query's.
            Some(slopes) if self.alibi_root => {
                let keys = attention.dim(candle_core::D::Minus1)?;
                let first_key = (index_pos + sequence) as f64 - keys as f64;
                let queries = (Tensor::arange(0u32, sequence as u32, attention.device())?
                    .to_dtype(DType::F32)?
                    + index_pos as f64)?
                    .reshape((1, 1, sequence, 1))?;
                let key_positions = (Tensor::arange(0u32, keys as u32, attention.device())?
                    .to_dtype(DType::F32)?
                    + first_key)?
                    .reshape((1, 1, 1, keys))?;
                let distance = queries.broadcast_sub(&key_positions)?.relu()?.sqrt()?;
                attention.broadcast_sub(&slopes.broadcast_mul(&distance)?)?
            }
            Some(slopes) => {
                let keys = attention.dim(candle_core::D::Minus1)?;
                let positions = Tensor::arange(0u32, keys as u32, attention.device())?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, 1, keys))?;
                attention.broadcast_add(&slopes.broadcast_mul(&positions)?)?
            }
            None => attention,
        };
        // Gemma 2 bounds every score to `(-cap, cap)` with a tanh before the
        // mask, so no single key can take the whole softmax.
        let attention = match self.softcap {
            Some(cap) => ((attention / cap)?.tanh()? * cap)?,
            None => attention,
        };
        let keys = attention.dim(candle_core::D::Minus1)?;
        let attention = match mask {
            // A supplied mask already carries the causal constraint, so it is
            // applied at every batch size instead of only when the query axis
            // is longer than one — a padded row must not attend to its own
            // filler however short the query axis happens to be.
            Some(mask) => {
                let mask = mask.broadcast_as(attention.shape())?;
                masked_fill(&attention, &mask, f32::NEG_INFINITY)?
            }
            // A lone query with no supplied mask can only reach keys that
            // already exist, so there is nothing causality would remove — and
            // nothing a window would either, while every key is inside it.
            None if sequence == 1 && window.is_none_or(|window| keys <= window) => attention,
            // A prompt read whole from position zero under a PrefixLM mask
            // is one bidirectional block. A pass that scores every position
            // is not a prompt, and stays causal.
            None if self.prefix_bidirectional
                && index_pos == 0
                && mode.readout != Readout::EveryPosition =>
            {
                attention
            }
            None => {
                let mask = cache
                    .mask(sequence, index_pos, window)?
                    .broadcast_as(attention.shape())?;
                masked_fill(&attention, &mask, f32::NEG_INFINITY)?
            }
        };
        // `softmax_last_dim` is `apply_op1_no_bwd` (candle-nn-0.11.0/src/ops.rs:438).
        // `ops::softmax` is the same softmax spelled out of `max_keepdim`,
        // `broadcast_sub`, `exp`, `sum_keepdim` and `broadcast_div`, all of which
        // record a backward node.
        // A sink adds one more logit per head to every query's softmax; its
        // share of the probability is dropped rather than spent on a value.
        let (attention, keys) = match &self.sinks {
            Some(sinks) => {
                let (batch, heads, queries, keys) = attention.dims4()?;
                let column = sinks.broadcast_as((batch, heads, queries, 1))?;
                (Tensor::cat(&[&attention, &column], 3)?, Some(keys))
            }
            None => (attention, None),
        };
        let attention = match mode.pass {
            Pass::Inference => candle_nn::ops::softmax_last_dim(&attention.contiguous()?)?,
            Pass::Differentiable => candle_nn::ops::softmax(&attention, candle_core::D::Minus1)?,
        };
        let attention = match keys {
            Some(keys) => attention.narrow(3, 0, keys)?,
            None => attention,
        };
        attention
            .matmul(&value.contiguous()?)?
            .to_dtype(input_dtype)
    }
}

/// The exponent range of ALiBi's geometric slopes: for `n` heads (a power of
/// two) the slopes are `2^(-span/n)` raised to `1..=n` (Press et al., 2022,
/// and Transformers' `build_alibi_tensor`).
pub(crate) const ALIBI_SPAN: f64 = 8.0;

/// ALiBi's slope for each of `heads` heads. A head count that is not a power
/// of two takes the slopes of the nearest power below, then every other slope
/// of twice that many, as BLOOM does.
pub(super) fn alibi_slopes(heads: usize) -> Vec<f32> {
    let closest = 1usize << heads.ilog2();
    let base = (-ALIBI_SPAN / closest as f64).exp2();
    let mut slopes: Vec<f32> = (1..=closest)
        .map(|power| base.powi(power as i32) as f32)
        .collect();
    if closest != heads {
        let extra = (-ALIBI_SPAN / (2 * closest) as f64).exp2();
        let remaining = closest.min(heads - closest);
        slopes.extend((0..remaining).map(|index| extra.powi((2 * index + 1) as i32) as f32));
    }
    slopes
}

fn repeat_key_value(input: Tensor, repeats: usize) -> candle_core::Result<Tensor> {
    if repeats == 1 {
        return Ok(input);
    }
    let (batch, key_value_heads, sequence, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((batch, key_value_heads, repeats, sequence, head_dim))?
        .reshape((batch, key_value_heads * repeats, sequence, head_dim))
}

pub(in crate::model) fn masked_fill(
    values: &Tensor,
    mask: &Tensor,
    replacement: f32,
) -> candle_core::Result<Tensor> {
    let replacement = Tensor::new(replacement, values.device())?.broadcast_as(mask.shape())?;
    mask.where_cond(&replacement, values)
}

/// The causal constraint and the key-padding constraint in one tensor.
///
/// A position is masked when it is in the query's future, `key > query`, or
/// when it is filler, `key >= lengths[row]`. The result is `[batch, 1,
/// sequence, sequence]`: one plane per row, and a unit head axis that
/// broadcasts, since every head of a row sees the same tokens.
///
/// Under the right padding [`SteeringLlama::forward_batch`] requires, the
/// second clause is redundant for a *real* query — `key <= query < length`
/// already implies `key < length` — and this says so rather than pretending
/// otherwise. What it buys is that the invariant holds by construction
/// instead of by coincidence of the padding side, and that a filler query,
/// whose row is computed whether or not anyone reads it, still mixes only
/// real keys. That second part is also why filler rows cannot produce a NaN:
/// row `query >= length` keeps keys `0..length`, which is never empty because
/// a zero length is refused.
///
/// On a sliding-window layer a real query also loses every key `window` or
/// more positions behind it, and on a chunked layer every key outside its
/// chunk and the `chunk_lookback` chunks before it. A filler query keeps
/// all real keys instead: a window past the end of its row could leave it
/// none, and a row of negative infinities is a NaN that a training step
/// would carry back into the adapters even though no loss reads that row.
///
/// Masked entries are `1`, matching [`Cache::mask`], so both feed the same
/// `masked_fill` and the same `f32::NEG_INFINITY`, which softmax turns into
/// exactly zero weight.
pub(in crate::model) fn padded_causal_mask(
    lengths: &[usize],
    sequence: usize,
    window: Option<usize>,
    chunk_lookback: Option<usize>,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut values = vec![0u8; lengths.len() * sequence * sequence];
    for (row, &length) in lengths.iter().enumerate() {
        let plane = row * sequence * sequence;
        for query in 0..sequence {
            let offset = plane + query * sequence;
            let visible = (query + 1).min(length);
            for slot in values[offset + visible..offset + sequence].iter_mut() {
                *slot = 1;
            }
            if query < length {
                for key in 0..visible {
                    if super::super::cache::hidden_key(query, key, window, chunk_lookback) {
                        values[offset + key] = 1;
                    }
                }
            }
        }
    }
    Tensor::from_vec(values, (lengths.len(), 1, sequence, sequence), device)
}
