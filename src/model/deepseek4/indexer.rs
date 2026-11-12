//! DeepSeek-V4's Lightning Indexer: which compressed entries a query on a
//! sparse layer keeps.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::super::{Cache, Pass, cache::RotaryTable};
use super::attention::rotate_trailing;
use super::compressor::Compressor;

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
        let keys = self
            .compressor
            .entries(hidden, layer, cache, rotation, pass)?
            .to_dtype(DType::F32)?;
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
        let scores = (queries
            .broadcast_matmul(&keys.unsqueeze(1)?.transpose(2, 3)?.contiguous()?)?
            .relu()?
            / (self.head_dim as f64).sqrt())?;
        let weights = (self.weights.forward(hidden)?.to_dtype(DType::F32)?
            / (self.heads as f64).sqrt())?
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
