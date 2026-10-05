//! DeepSeek-V4's compressors (`DeepseekV4HCACompressor`,
//! `DeepseekV4CSACompressor` and the indexer's own).
//!
//! A compressor turns every window of `rate` tokens into one entry: each
//! token's `kv_proj(x)` weighted by the softmax over the window of its
//! `gate_proj(x) + position_bias`, summed, RMS-normed (`kv_norm`) and rotated
//! at the window's first position. An overlapping compressor (the sparse
//! layers' and the indexer's) projects two series per token: an entry mixes
//! its own window's second series with the window before's first, over
//! `2 · rate` slots; the window before the first of a call is the last of
//! the call before, kept in the state. Tokens past the last whole window
//! wait in the state for the next call.

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::super::{
    Cache, NormKind, Pass,
    cache::RotaryTable,
    layer::norm::{Norm, NormSpec},
};
use super::attention::rotate_trailing;

/// One compressor's running state across the calls of a decode.
#[derive(Debug, Clone, Default)]
pub(in crate::model) struct CompressorState {
    /// Projected tokens past the last whole window: `kv` and `gate`.
    pending: Option<(Tensor, Tensor)>,
    /// Every entry made so far, `[batch, entries, width]`.
    entries: Option<Tensor>,
    count: usize,
    /// The last window's first series, `kv` and `gate`, `[batch, rate,
    /// width]`.
    overlap: Option<(Tensor, Tensor)>,
}

#[derive(Debug, Clone)]
pub(super) struct Compressor {
    key_value: Linear,
    gate: Linear,
    /// `[rate, series · width]`.
    position_bias: Tensor,
    norm: Norm,
    rate: usize,
    width: usize,
    overlapping: bool,
    slot: usize,
}

impl Compressor {
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        width: usize,
        rate: usize,
        overlapping: bool,
        eps: f64,
        slot: usize,
    ) -> candle_core::Result<Self> {
        let series = if overlapping { 2 } else { 1 };
        Ok(Self {
            key_value: linear_no_bias(hidden, series * width, builder.pp("kv_proj"))?,
            gate: linear_no_bias(hidden, series * width, builder.pp("gate_proj"))?,
            position_bias: builder.get((rate, series * width), "position_bias")?,
            norm: NormSpec { kind: NormKind::Rms, eps, offset: false, groups: 1 }.load(width, builder.pp("kv_norm"))?,
            rate,
            width,
            overlapping,
            slot,
        })
    }

    pub(super) fn rate(&self) -> usize {
        self.rate
    }

    /// Every entry made so far, `[batch, entries, width]`, after reading
    /// `hidden` `[batch, sequence, hidden]`.
    pub(super) fn entries(
        &self,
        hidden: &Tensor,
        layer: usize,
        cache: &mut Cache,
        rotation: &RotaryTable,
        pass: Pass,
    ) -> candle_core::Result<Tensor> {
        let (batch, _, _) = hidden.dims3()?;
        let key_value = self.key_value.forward(hidden)?;
        let gate = self.gate.forward(hidden)?;
        let keep = cache.use_kv_cache;
        let mut state = if keep { cache.compressors.remove(&(layer, self.slot)).unwrap_or_default() } else { CompressorState::default() };
        let first = state.count * self.rate;
        let (key_value, gate) = match state.pending.take() {
            Some((pending_kv, pending_gate)) => {
                (Tensor::cat(&[&pending_kv, &key_value], 1)?, Tensor::cat(&[&pending_gate, &gate], 1)?)
            }
            None => (key_value, gate),
        };
        let length = key_value.dim(1)?;
        let usable = length / self.rate * self.rate;
        if keep && usable < length {
            state.pending = Some((
                key_value.narrow(1, usable, length - usable)?.contiguous()?,
                gate.narrow(1, usable, length - usable)?.contiguous()?,
            ));
        }
        let windows = usable / self.rate;
        if windows > 0 {
            let series = if self.overlapping { 2 } else { 1 };
            let shape = (batch, windows, self.rate, series * self.width);
            let key_value = key_value.narrow(1, 0, usable)?.reshape(shape)?;
            let gate = gate.narrow(1, 0, usable)?.reshape(shape)?.broadcast_add(&self.position_bias)?;
            let (slots_kv, slots_gate) = if self.overlapping {
                self.overlapped(&key_value, &gate, keep.then_some(&mut state))?
            } else {
                (key_value, gate)
            };
            let weights = candle_nn::ops::softmax(&slots_gate.to_dtype(DType::F32)?, 2)?.to_dtype(slots_kv.dtype())?;
            let pooled = (slots_kv * weights)?.sum(2)?;
            let normed = self.norm.forward(&pooled, pass)?;
            let positions: Vec<usize> = (0..windows).map(|window| first + window * self.rate).collect();
            let (cos, sin) = rotation.angles_at(&positions)?;
            let rotated = rotate_trailing(&normed.unsqueeze(1)?, &cos, &sin, pass)?.squeeze(1)?;
            state.entries = Some(match state.entries.take() {
                Some(earlier) => Tensor::cat(&[&earlier, &rotated], 1)?,
                None => rotated,
            });
            state.count += windows;
        }
        let entries = match &state.entries {
            Some(entries) => entries.clone(),
            None => Tensor::zeros((batch, 0, self.width), hidden.dtype(), hidden.device())?,
        };
        if keep {
            cache.compressors.insert((layer, self.slot), state);
        }
        Ok(entries)
    }

    /// The `2 · rate` slots of every window: the window before's first
    /// series, then its own second; the first window's earlier half from
    /// the state, or empty (zero, gated to nothing) on a decode's first
    /// window. The last window's first series is kept for the next call.
    fn overlapped(
        &self,
        key_value: &Tensor,
        gate: &Tensor,
        state: Option<&mut CompressorState>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (batch, windows, rate, _) = key_value.dims4()?;
        let width = self.width;
        let first_kv = key_value.narrow(3, 0, width)?;
        let first_gate = gate.narrow(3, 0, width)?;
        let prior = match state {
            Some(state) => {
                let prior = state.overlap.take();
                state.overlap = Some((
                    first_kv.narrow(1, windows - 1, 1)?.squeeze(1)?.contiguous()?,
                    first_gate.narrow(1, windows - 1, 1)?.squeeze(1)?.contiguous()?,
                ));
                prior
            }
            None => None,
        };
        let (prior_kv, prior_gate) = match prior {
            Some((kv, gate)) => (kv.unsqueeze(1)?, gate.unsqueeze(1)?),
            None => (
                Tensor::zeros((batch, 1, rate, width), key_value.dtype(), key_value.device())?,
                Tensor::full(f32::NEG_INFINITY, (batch, 1, rate, width), key_value.device())?.to_dtype(gate.dtype())?,
            ),
        };
        let earlier = |prior: &Tensor, series: &Tensor| -> candle_core::Result<Tensor> {
            if windows > 1 {
                Tensor::cat(&[prior, &series.narrow(1, 0, windows - 1)?], 1)
            } else {
                Ok(prior.clone())
            }
        };
        let slots_kv = Tensor::cat(&[&earlier(&prior_kv, &first_kv)?, &key_value.narrow(3, width, width)?], 2)?;
        let slots_gate = Tensor::cat(&[&earlier(&prior_gate, &first_gate)?, &gate.narrow(3, width, width)?], 2)?;
        Ok((slots_kv.contiguous()?, slots_gate.contiguous()?))
    }
}

/// Which entries each query may see, `[batch, sequence, entries]` as 1
/// for hidden: an entry is visible once its window has closed at or before
/// the query, `entry < (position + 1) / rate`.
pub(super) fn closed_entries(
    index_pos: usize,
    sequence: usize,
    entries: usize,
    rate: usize,
    device: &candle_core::Device,
) -> candle_core::Result<Tensor> {
    let mut hidden = Vec::with_capacity(sequence * entries);
    for query in 0..sequence {
        let threshold = (index_pos + query + 1) / rate;
        hidden.extend((0..entries).map(|entry| u8::from(entry >= threshold)));
    }
    Tensor::from_vec(hidden, (1, sequence, entries), device)
}

/// The sparse layers' Lightning Indexer: queries from the latent query
/// (`q_b_proj`), keys from its own overlapping compressor at
/// `index_head_dim`; a query scores an entry `Σ_h w_h · relu(q_h · k)` and
/// keeps its best `index_topk` among the entries closed before it.
#[derive(Debug, Clone)]
pub(super) struct Indexer {
    compressor: Compressor,
    query: Linear,
    weights: Linear,
    heads: usize,
    head_dim: usize,
    top_k: usize,
}

impl Indexer {
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        query_rank: usize,
        (heads, head_dim, top_k, rate): (usize, usize, usize, usize),
        eps: f64,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            compressor: Compressor::load(builder, hidden, head_dim, rate, true, eps, 1)?,
            query: linear_no_bias(query_rank, heads * head_dim, builder.pp("q_b_proj"))?,
            weights: linear_no_bias(hidden, heads, builder.pp("scorer.weights_proj"))?,
            heads,
            head_dim,
            top_k,
        })
    }

    /// Which of `entries` entries each query keeps, `[batch, sequence,
    /// entries]` as 1 for hidden.
    pub(super) fn hidden_entries(
        &self,
        hidden: &Tensor,
        latent_query: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        rotation: &RotaryTable,
        pass: Pass,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let keys = self.compressor.entries(hidden, layer, cache, rotation, pass)?.to_dtype(DType::F32)?;
        let entries = keys.dim(1)?;
        let rate = self.compressor.rate();
        let device = hidden.device();
        let queries = self
            .query
            .forward(latent_query)?
            .reshape((batch, sequence, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (cos, sin) = rotation.angles(index_pos, sequence)?;
        let queries = rotate_trailing(&queries, &cos, &sin, pass)?.to_dtype(DType::F32)?;
        // `[batch, heads, sequence, entries]`, then summed over heads.
        let scores = (queries.broadcast_matmul(&keys.unsqueeze(1)?.transpose(2, 3)?.contiguous()?)?.relu()?
            / (self.head_dim as f64).sqrt())?;
        let weights = (self.weights.forward(hidden)?.to_dtype(DType::F32)? / (self.heads as f64).sqrt())?
            .transpose(1, 2)?
            .unsqueeze(3)?;
        let scores = scores.broadcast_mul(&weights)?.sum(1)?.to_vec3::<f32>()?;
        let mut hidden_entries = vec![1u8; batch * sequence * entries];
        for (row_batch, rows) in scores.iter().enumerate() {
            for (query, row) in rows.iter().enumerate() {
                let threshold = ((index_pos + query + 1) / rate).min(entries);
                let mut order: Vec<usize> = (0..threshold).collect();
                order.sort_by(|left, right| row[*right].total_cmp(&row[*left]));
                for entry in order.into_iter().take(self.top_k) {
                    hidden_entries[(row_batch * sequence + query) * entries + entry] = 0;
                }
            }
        }
        Tensor::from_vec(hidden_entries, (batch, sequence, entries), device)
    }
}
