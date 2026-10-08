//! MiniMax Sparse Attention's block indexer (MiniMax-M3): which blocks of
//! keys each query attends to, chosen apart for every key-value group.
//!
//! The index branch projects the hidden state to one query per key-value
//! group (`index_q_proj`, normed per head by `index_q_norm`) and one key
//! shared by all of them (`index_k_proj`, `index_k_norm`), both rotated by
//! the main attention's angles over its rotated share. A key scores as the
//! plain product of the two, unscaled; a block of keys scores as its best key
//! the query may see, and each query keeps its group's best blocks, its own
//! block and the local ones before it always among them. Blocks are counted
//! from the first key slot. This is Transformers' `MiniMaxM3VLIndexer`
//! (`modeling_minimax_m3_vl.py`), which vLLM's `MiniMaxM3Indexer` runs as
//! kernels. While no more blocks exist than a query keeps, every key is kept
//! and the attention is the dense one; the index keys are still cached for
//! later calls.

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::layer::norm::{Norm, NormSpec};
use crate::model::{BlockIndexSpec, Cache, Mode};

#[derive(Debug, Clone)]
pub(super) struct BlockIndexer {
    query: Linear,
    key: Linear,
    query_norm: Norm,
    key_norm: Norm,
    spec: BlockIndexSpec,
}

impl BlockIndexer {
    /// `builder` is the attention block's (`self_attn`): the index branch's
    /// projections and norms sit beside the main ones.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        norm: NormSpec,
        spec: BlockIndexSpec,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            query: linear_no_bias(hidden, spec.heads * spec.head_dim, builder.pp("index_q_proj"))?,
            key: linear_no_bias(hidden, spec.head_dim, builder.pp("index_k_proj"))?,
            query_norm: norm.load(spec.head_dim, builder.pp("index_q_norm"))?,
            key_norm: norm.load(spec.head_dim, builder.pp("index_k_norm"))?,
            spec,
        })
    }

    /// The keys each of the attention's `heads` query heads may not see
    /// beyond causality, `[batch, heads, sequence, keys]` as one for hidden,
    /// or `None` while every block is kept. `angles` are the main
    /// attention's for these positions, which start at `index_pos`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hidden_keys(
        &self,
        hidden: &Tensor,
        angles: Option<&(Tensor, Tensor)>,
        heads: usize,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<Option<Tensor>> {
        let (batch, sequence, _) = hidden.dims3()?;
        let BlockIndexSpec {
            heads: groups,
            head_dim,
            block,
            top_blocks,
            local_blocks,
            ..
        } = self.spec;
        let block = block.get();
        // The rotation's angles on the host, one row per position.
        let angles = match angles {
            Some((cos, sin)) => Some((
                cos.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                sin.to_dtype(DType::F32)?.to_vec2::<f32>()?,
            )),
            None => None,
        };
        // `[batch, sequence, heads, head_dim]` normed, rotated, laid out as
        // `[batch, sequence · heads, head_dim]`.
        let branch = |projection: &Linear,
                      norm: &Norm,
                      branch_heads: usize|
         -> candle_core::Result<Tensor> {
            let projected = projection
                .forward(hidden)?
                .reshape((batch, sequence, branch_heads, head_dim))?;
            let mut values = norm
                .forward(&projected, mode.pass)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if let Some(angles) = &angles {
                rotate(&mut values, head_dim, branch_heads, sequence, angles)?;
            }
            Tensor::from_vec(
                values,
                (batch, sequence * branch_heads, head_dim),
                hidden.device(),
            )
        };
        // The index key every group shares, `[batch, keys, head_dim]`: as
        // many heads as `index_k_proj` stores rows of `head_dim`.
        let key_heads = self.key.weight().dim(D::Minus2)? / head_dim;
        let mut key = branch(&self.key, &self.key_norm, key_heads)?;
        if cache.use_kv_cache {
            if let Some(earlier) = &cache.index_keys[layer] {
                key = Tensor::cat(&[earlier, &key], D::Minus2)?.contiguous()?;
            }
            cache.index_keys[layer] = Some(key.clone());
        }
        let keys = key.dim(D::Minus2)?;
        if keys.div_ceil(block) <= top_blocks {
            return Ok(None);
        }
        let repeats = heads / groups;
        if repeats * groups != heads {
            candle_core::bail!(
                "layer {layer} has {heads} query heads, which its {groups} index heads do not divide into equal groups"
            );
        }
        let query = branch(&self.query, &self.query_norm, groups)?;
        // `[batch, sequence · groups, keys]` scores, unscaled.
        let scores = query.matmul(&key.t()?.contiguous()?)?.to_vec3::<f32>()?;
        let mut visible = vec![false; batch * heads * sequence * keys];
        for (sample, rows) in scores.iter().enumerate() {
            for (row, row_scores) in rows.iter().enumerate() {
                let (offset, group) = (row / groups, row % groups);
                let position = index_pos + offset;
                let own = position / block;
                // Each block's best key at or before this query; a block
                // wholly after it scores minus infinity and is never kept.
                // The query's own block and the local ones before it are
                // always kept.
                let mut ranked: Vec<(usize, f32)> = row_scores
                    .chunks(block)
                    .enumerate()
                    .map(|(chosen, members)| {
                        let start = chosen * block;
                        let best = members
                            .iter()
                            .enumerate()
                            .filter(|(member, _)| start + member <= position)
                            .map(|(_, score)| *score)
                            .fold(f32::NEG_INFINITY, f32::max);
                        if chosen <= own && chosen + local_blocks > own {
                            (chosen, f32::INFINITY)
                        } else {
                            (chosen, best)
                        }
                    })
                    .collect();
                ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
                let kept = ranked
                    .into_iter()
                    .take(top_blocks)
                    .filter(|(_, score)| *score != f32::NEG_INFINITY);
                for (chosen, _) in kept {
                    let start = chosen * block;
                    for key in start..(start + block).min(keys) {
                        for head in group * repeats..group * repeats + repeats {
                            visible[((sample * heads + head) * sequence + offset) * keys + key] =
                                true;
                        }
                    }
                }
            }
        }
        let hidden_keys: Vec<u8> = visible.into_iter().map(|seen| u8::from(!seen)).collect();
        Tensor::from_vec(hidden_keys, (batch, heads, sequence, keys), hidden.device()).map(Some)
    }
}

/// Rotate every `head_dim`-wide head of `values`, laid out as `[batch,
/// sequence, heads, head_dim]`, by its position's angles: the first half
/// against the second half of the rotated share, as Transformers'
/// `rotate_half` does, and the rest of the head passed through.
fn rotate(
    values: &mut [f32],
    head_dim: usize,
    heads: usize,
    sequence: usize,
    (cos, sin): &(Vec<Vec<f32>>, Vec<Vec<f32>>),
) -> candle_core::Result<()> {
    for (index, head) in values.chunks_mut(head_dim).enumerate() {
        let offset = (index / heads) % sequence;
        let (cos, sin) = (&cos[offset], &sin[offset]);
        let half = cos.len();
        if half + half > head_dim {
            candle_core::bail!(
                "the index heads of {head_dim} components cannot rotate {} of them",
                half + half
            );
        }
        let (first, rest) = head.split_at_mut(half);
        for (((leading, trailing), cos), sin) in first
            .iter_mut()
            .zip(rest[..half].iter_mut())
            .zip(cos)
            .zip(sin)
        {
            let (former, latter) = (*leading, *trailing);
            *leading = former * cos - latter * sin;
            *trailing = latter * cos + former * sin;
        }
    }
    Ok(())
}
