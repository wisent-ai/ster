//! DeepSeek Sparse Attention's indexer (DeepSeek-V3.2, GLM-5): which keys
//! each query attends to once a layer sees more of them than `index_topk`.
//!
//! The indexer scores every key for every query with its own lightweight
//! projections — `wq_b` over the latent query `q_a_layernorm(q_a_proj(x))`,
//! `wk` and the LayerNorm `k_norm` over the hidden state, the first
//! `qk_rope_head_dim` components of each rotated — as
//! `Σ_h w_h · relu(q_h · k / sqrt(index_head_dim))`, with
//! `w = weights_proj(x) / sqrt(index_n_heads)`, and keeps each query's top
//! `index_topk` keys; the attention hides the rest. With no more keys than
//! that, every key is kept and the attention is the dense one, so the scores
//! are skipped; the indexer's keys are still cached for later calls.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{Cache, IndexerSpec, KeyPool, Mode, NormKind};

use super::apply_rotary;
use crate::model::layer::norm::{Norm, NormSpec};

/// The epsilon of the indexer's key norm, which it fixes rather than reading
/// the model's (`nn.LayerNorm(self.head_dim, eps=1e-6)` in Transformers'
/// `DeepseekV32Indexer`).
const KEY_NORM_EPS: f64 = 1e-6;

#[derive(Debug, Clone)]
pub(super) struct Indexer {
    query: Linear,
    key: Linear,
    key_norm: Norm,
    weights: Linear,
    rotated: usize,
    spec: IndexerSpec,
    /// GLM-5-Next's pool weighting: `index_kpool_compress_gate` and
    /// `index_kpool_compress_ape` `[pool, head_dim]` in F32.
    pool: Option<(Linear, Tensor)>,
}

impl Indexer {
    /// `builder` is the attention block's; the indexer sits below it
    /// (`self_attn.indexer`). `query_rank` is the latent query's width and
    /// `rotated` the rotated share of each head.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        query_rank: usize,
        rotated: usize,
        spec: IndexerSpec,
    ) -> candle_core::Result<Self> {
        let block = builder.pp("indexer");
        let norm = NormSpec {
            kind: NormKind::Layer { bias: true },
            eps: KEY_NORM_EPS,
            offset: false,
            groups: 1,
        };
        Ok(Self {
            query: linear_no_bias(query_rank, spec.heads * spec.head_dim, block.pp("wq_b"))?,
            key: linear_no_bias(hidden, spec.head_dim, block.pp("wk"))?,
            key_norm: norm.load(spec.head_dim, block.pp("k_norm"))?,
            weights: linear_no_bias(hidden, spec.heads, block.pp("weights_proj"))?,
            rotated,
            pool: match spec.pool {
                Some(pool) => Some((
                    Linear::new(
                        block.get((spec.head_dim, hidden), "index_kpool_compress_gate")?,
                        None,
                    ),
                    block
                        .get((pool.size, spec.head_dim), "index_kpool_compress_ape")?
                        .to_dtype(DType::F32)?,
                )),
                None => None,
            },
            spec,
        })
    }

    /// The keys this call hides beyond causality, `[batch, 1, sequence,
    /// keys]` as 1 for hidden, or `None` when every key is kept. `causal` is
    /// the causal (and padding) constraint, broadcastable to `[batch,
    /// sequence, keys]`, 1 where hidden; `angles` are the main attention's
    /// for these positions.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hidden_keys(
        &self,
        hidden: &Tensor,
        latent_query: &Tensor,
        angles: Option<&(Tensor, Tensor)>,
        causal: &Tensor,
        layer: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<Option<Tensor>> {
        let (batch, sequence, _) = hidden.dims3()?;
        let IndexerSpec {
            heads,
            head_dim,
            top_k,
            interleaved,
            ..
        } = self.spec;
        let rotate = |input: Tensor| -> candle_core::Result<Tensor> {
            match angles {
                Some((cos, sin)) => {
                    let width = cos.dim(1)? * 2;
                    if width != self.rotated {
                        candle_core::bail!(
                            "the indexer rotates {} components, and this layer's angles span {width}",
                            self.rotated
                        );
                    }
                    apply_rotary(&input, cos, sin, DType::F32, mode.pass, interleaved)
                }
                None => input.to_dtype(DType::F32),
            }
        };
        let key = self
            .key_norm
            .forward(&self.key.forward(hidden)?, mode.pass)?;
        let mut key = rotate(key.reshape((batch, 1, sequence, head_dim))?)?;
        // A pooled indexer keeps each key's pool gate beside it.
        if let Some((gate, _)) = &self.pool {
            let gates = gate
                .forward(hidden)?
                .to_dtype(DType::F32)?
                .reshape((batch, 1, sequence, head_dim))?;
            key = Tensor::cat(&[&key, &gates], 3)?;
        }
        if cache.use_kv_cache {
            if let Some(earlier) = &cache.index_keys[layer] {
                key = Tensor::cat(&[earlier, &key], 2)?.contiguous()?;
            }
            cache.index_keys[layer] = Some(key.clone());
        }
        let keys = key.dim(2)?;
        let query = || -> candle_core::Result<Tensor> {
            let query = self
                .query
                .forward(latent_query)?
                .reshape((batch, sequence, heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            rotate(query)
        };
        let weights = || -> candle_core::Result<Tensor> {
            (self.weights.forward(hidden)?.to_dtype(DType::F32)? / (heads as f64).sqrt())?
                .transpose(1, 2)?
                .reshape((batch, heads, sequence, 1))
        };
        if let (Some((_, position_bias)), Some(pool)) = (&self.pool, self.spec.pool) {
            let index_pos = keys - sequence;
            let hidden_keys = pooled(
                &key,
                position_bias,
                query()?,
                weights()?,
                pool,
                top_k,
                index_pos,
            )?;
            return Ok(Some(hidden_keys));
        }
        if keys <= top_k {
            return Ok(None);
        }
        let query = query()?;
        let scores = (query.matmul(&key.t()?)? / (head_dim as f64).sqrt())?.relu()?;
        let weights = weights()?;
        let index = scores.broadcast_mul(&weights)?.sum(1)?;
        let causal = causal.broadcast_as((batch, sequence, keys))?;
        let index = causal.where_cond(
            &Tensor::full(f32::NEG_INFINITY, (batch, sequence, keys), index.device())?,
            &index,
        )?;
        let chosen = index
            .contiguous()?
            .arg_sort_last_dim(false)?
            .narrow(2, 0, top_k)?
            .contiguous()?;
        let kept = Tensor::full(-1f32, (batch, sequence, top_k), index.device())?;
        let hidden_keys = Tensor::ones((batch, sequence, keys), DType::F32, index.device())?
            .scatter_add(&chosen, &kept, 2)?;
        hidden_keys.to_dtype(DType::U8)?.unsqueeze(1).map(Some)
    }
}

/// The keys a pooled indexer hides, `[batch, 1, sequence, keys]` as 1 for
/// hidden: everything but each query's chosen pools and, under `tail`, its
/// own unfinished pool. `packed` holds every key and its pool gate side by
/// side, `[batch, 1, keys, 2 · head_dim]`; queries start at `index_pos`.
fn pooled(
    packed: &Tensor,
    position_bias: &Tensor,
    query: Tensor,
    weights: Tensor,
    pool: KeyPool,
    top_k: usize,
    index_pos: usize,
) -> candle_core::Result<Tensor> {
    let (batch, _, sequence, head_dim) = query.dims4()?;
    let keys = packed.dim(2)?;
    let device = packed.device();
    let pools = keys / pool.size;
    let chosen_per_query = (top_k / pool.size).min(pools);
    // `[batch, sequence, pools]` scores, empty when no pool has closed.
    let scores = if pools > 0 {
        let whole = packed
            .narrow(2, 0, pools * pool.size)?
            .squeeze(1)?
            .reshape((batch, pools, pool.size, 2 * head_dim))?;
        let pool_keys = whole.narrow(3, 0, head_dim)?;
        let gates = whole
            .narrow(3, head_dim, head_dim)?
            .broadcast_add(position_bias)?;
        let mix = candle_nn::ops::softmax(&gates, 2)?;
        let pool_keys = (pool_keys * mix)?.sum(2)?.unsqueeze(1)?;
        let scores = (query.broadcast_matmul(&pool_keys.transpose(2, 3)?.contiguous()?)?
            / (head_dim as f64).sqrt())?
        .relu()?;
        scores.broadcast_mul(&weights)?.sum(1)?.to_vec3::<f32>()?
    } else {
        vec![vec![Vec::new(); sequence]; batch]
    };
    let mut hidden = vec![1u8; batch * sequence * keys];
    for (row, rows) in scores.iter().enumerate() {
        for (offset, row_scores) in rows.iter().enumerate() {
            let position = index_pos + offset;
            let base = (row * sequence + offset) * keys;
            // The pools that ended at or before this query, best first.
            let closed = ((position + 1) / pool.size).min(pools);
            let mut order: Vec<usize> = (0..closed).collect();
            order.sort_by(|left, right| row_scores[*right].total_cmp(&row_scores[*left]));
            for chosen in order.into_iter().take(chosen_per_query) {
                for key in chosen * pool.size..(chosen + 1) * pool.size {
                    hidden[base + key] = 0;
                }
            }
            if pool.tail {
                let unfinished = (position + 1) % pool.size;
                for key in position + 1 - unfinished..=position {
                    hidden[base + key] = 0;
                }
            }
        }
    }
    Tensor::from_vec(hidden, (batch, 1, sequence, keys), device)
}
