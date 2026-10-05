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

use crate::model::{Cache, IndexerSpec, Mode, NormKind};

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
        let norm = NormSpec { kind: NormKind::Layer { bias: true }, eps: KEY_NORM_EPS, offset: false, groups: 1 };
        Ok(Self {
            query: linear_no_bias(query_rank, spec.heads * spec.head_dim, block.pp("wq_b"))?,
            key: linear_no_bias(hidden, spec.head_dim, block.pp("wk"))?,
            key_norm: norm.load(spec.head_dim, block.pp("k_norm"))?,
            weights: linear_no_bias(hidden, spec.heads, block.pp("weights_proj"))?,
            rotated,
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
        let IndexerSpec { heads, head_dim, top_k, interleaved, .. } = self.spec;
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
        let key = self.key_norm.forward(&self.key.forward(hidden)?, mode.pass)?;
        let mut key = rotate(key.reshape((batch, 1, sequence, head_dim))?)?;
        if cache.use_kv_cache {
            if let Some(earlier) = &cache.index_keys[layer] {
                key = Tensor::cat(&[earlier, &key], 2)?.contiguous()?;
            }
            cache.index_keys[layer] = Some(key.clone());
        }
        let keys = key.dim(2)?;
        if keys <= top_k {
            return Ok(None);
        }
        let query = self
            .query
            .forward(latent_query)?
            .reshape((batch, sequence, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query = rotate(query)?;
        let scores = (query.matmul(&key.t()?)? / (head_dim as f64).sqrt())?.relu()?;
        let weights = (self.weights.forward(hidden)?.to_dtype(DType::F32)? / (heads as f64).sqrt())?
            .transpose(1, 2)?
            .reshape((batch, heads, sequence, 1))?;
        let index = scores.broadcast_mul(&weights)?.sum(1)?;
        let causal = causal.broadcast_as((batch, sequence, keys))?;
        let index = causal.where_cond(&Tensor::full(f32::NEG_INFINITY, (batch, sequence, keys), index.device())?, &index)?;
        let chosen = index.contiguous()?.arg_sort_last_dim(false)?.narrow(2, 0, top_k)?.contiguous()?;
        let kept = Tensor::full(-1f32, (batch, sequence, top_k), index.device())?;
        let hidden_keys = Tensor::ones((batch, sequence, keys), DType::F32, index.device())?.scatter_add(&chosen, &kept, 2)?;
        hidden_keys.to_dtype(DType::U8)?.unsqueeze(1).map(Some)
    }
}
