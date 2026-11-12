//! Inkling's attention (`InklingAttention` in `modeling_inkling.py`, below
//! `attn` in Thinking Machines' layout): no rotation; each query's scores
//! carry a bias by its distance to the key, which the query's own
//! `wr_du(x)`, `d_rel` wide per head, mixes from the layer's bank of
//! bias-versus-distance profiles `rel_logits_proj.proj` `[d_rel, extent]`,
//! and which is zero at a distance of `extent` or more. Queries and keys are
//! RMS-normed per head and the scores divided by the head width; keys and
//! values pass their residual short convolutions (`k_sconv`, `v_sconv`)
//! before the norm; the output passes `attn_sconv`. On the full layers,
//! under `log_scaling_n_floor`, the scores and the bias at position `p` are
//! multiplied by `1 + alpha · ln(max((p + 1) / floor, 1))`.

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::super::{
    Cache, InklingSpec, Mode, NormKind, Pass,
    inkling::ResidualConv,
    layer::norm::{Norm, NormSpec},
};

#[derive(Debug, Clone)]
pub(in crate::model) struct RelativeAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    relative: Linear,
    output: Linear,
    key_conv: ResidualConv,
    value_conv: ResidualConv,
    output_conv: ResidualConv,
    query_norm: Norm,
    key_norm: Norm,
    /// `[d_rel, extent]` in F32.
    profiles: Tensor,
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    profile_count: usize,
    extent: usize,
    window: Option<usize>,
    log_scaling: Option<(f64, f64)>,
}

impl RelativeAttention {
    /// `builder` is the layer's.
    pub(in crate::model) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: &InklingSpec,
        eps: f64,
        layer: usize,
    ) -> candle_core::Result<Self> {
        let sliding = spec.sliding_layers >> layer & 1 == 1;
        let shape = if sliding { spec.sliding } else { spec.full };
        let block = builder.pp("attn");
        let (queries, keys) = (
            shape.heads * shape.head_dim,
            shape.key_value_heads * shape.head_dim,
        );
        let norm = NormSpec {
            kind: NormKind::Rms,
            eps,
            offset: false,
            groups: 1,
        };
        Ok(Self {
            query: linear_no_bias(hidden, queries, block.pp("wq_du"))?,
            key: linear_no_bias(hidden, keys, block.pp("wk_dv"))?,
            value: linear_no_bias(hidden, keys, block.pp("wv_dv"))?,
            relative: linear_no_bias(hidden, shape.heads * spec.profiles, block.pp("wr_du"))?,
            output: linear_no_bias(queries, hidden, block.pp("wo_ud"))?,
            key_conv: ResidualConv::load(&block, "k_sconv", keys, spec.kernel, 0)?,
            value_conv: ResidualConv::load(&block, "v_sconv", keys, spec.kernel, 1)?,
            output_conv: ResidualConv::load(builder, "attn_sconv", hidden, spec.kernel, 2)?,
            query_norm: norm.load(shape.head_dim, block.pp("q_norm"))?,
            key_norm: norm.load(shape.head_dim, block.pp("k_norm"))?,
            profiles: block
                .pp("rel_logits_proj")
                .get((spec.profiles, shape.extent), "proj")?
                .to_dtype(DType::F32)?,
            heads: shape.heads,
            key_value_heads: shape.key_value_heads,
            head_dim: shape.head_dim,
            profile_count: spec.profiles,
            extent: shape.extent,
            window: sliding.then_some(shape.extent),
            log_scaling: if sliding { None } else { spec.log_scaling },
        })
    }

    /// The window this layer attends through, on a sliding layer.
    pub(in crate::model) fn window(&self) -> Option<usize> {
        self.window
    }

    /// `mask`, when given, hides a key wherever it is non-zero, as the
    /// decoder's padded masks do; otherwise the causal (and window) mask is
    /// the cache's.
    pub(in crate::model) fn forward(
        &self,
        normed: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = normed.dims3()?;
        let dtype = normed.dtype();
        let heads =
            |input: Tensor, count: usize| input.reshape((batch, sequence, count, self.head_dim));
        let query = self
            .query_norm
            .forward(&heads(self.query.forward(normed)?, self.heads)?, mode.pass)?;
        let key = self
            .key_conv
            .forward(&self.key.forward(normed)?, layer, cache)?;
        let key = self
            .key_norm
            .forward(&heads(key, self.key_value_heads)?, mode.pass)?;
        let value = heads(
            self.value_conv
                .forward(&self.value.forward(normed)?, layer, cache)?,
            self.key_value_heads,
        )?;
        let mut key = key.transpose(1, 2)?.contiguous()?;
        let mut value = value.transpose(1, 2)?.contiguous()?;
        if cache.use_kv_cache {
            if let Some((cached_key, cached_value)) = &cache.kvs[layer] {
                key = Tensor::cat(&[cached_key, &key], 2)?.contiguous()?;
                value = Tensor::cat(&[cached_value, &value], 2)?.contiguous()?;
            }
            cache.kvs[layer] = Some((key.clone(), value.clone()));
        }
        let total = key.dim(2)?;
        let first_key = (index_pos + sequence) as i64 - total as i64;
        let repeats = self.heads / self.key_value_heads;
        let repeat = |input: Tensor| -> candle_core::Result<Tensor> {
            if repeats == 1 {
                return Ok(input);
            }
            let (batch, groups, keys, width) = input.dims4()?;
            input
                .unsqueeze(2)?
                .expand((batch, groups, repeats, keys, width))?
                .reshape((batch, groups * repeats, keys, width))
        };
        let query = query.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
        let key = repeat(key)?.to_dtype(DType::F32)?;
        let value = repeat(value)?.to_dtype(DType::F32)?;
        let scores = (query.matmul(&key.t()?.contiguous()?)? / self.head_dim as f64)?;
        // Each head's bias at every distance, `[batch, heads, sequence,
        // extent]`, and the key at each distance gathered from it.
        let relative = self
            .relative
            .forward(normed)?
            .to_dtype(DType::F32)?
            .reshape((batch, sequence, self.heads, self.profile_count))?
            .matmul(
                &self
                    .profiles
                    .broadcast_left((batch, sequence))?
                    .contiguous()?,
            )?
            .transpose(1, 2)?
            .contiguous()?;
        let mut indices = Vec::with_capacity(sequence * total);
        let mut inside = Vec::with_capacity(sequence * total);
        for query_index in 0..sequence {
            let position = (index_pos + query_index) as i64;
            for key_index in 0..total {
                let distance = position - (first_key + key_index as i64);
                let within = (0..self.extent as i64).contains(&distance);
                indices.push(distance.clamp(0, self.extent as i64 - 1) as u32);
                inside.push(if within { 1f32 } else { 0f32 });
            }
        }
        let device = normed.device();
        let indices = Tensor::from_vec(indices, (1, 1, sequence, total), device)?
            .broadcast_as((batch, self.heads, sequence, total))?
            .contiguous()?;
        let inside = Tensor::from_vec(inside, (1, 1, sequence, total), device)?;
        let bias = relative
            .gather(&indices, D::Minus1)?
            .broadcast_mul(&inside)?;
        let scores = (scores + bias)?;
        let scores = match self.log_scaling {
            Some((alpha, floor)) => {
                let scales: Vec<f32> = (index_pos..index_pos + sequence)
                    .map(|position| {
                        (1.0 + alpha * ((position + 1) as f64 / floor).max(1.0).ln()) as f32
                    })
                    .collect();
                scores.broadcast_mul(&Tensor::from_vec(scales, (1, 1, sequence, 1), device)?)?
            }
            None => scores,
        };
        let hidden = match mask {
            Some(mask) => mask.broadcast_as(scores.shape())?.to_dtype(DType::U8)?,
            None => cache
                .mask(sequence, index_pos, self.window)?
                .broadcast_as(scores.shape())?,
        };
        let floor = Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as(scores.shape())?;
        let scores = hidden.where_cond(&floor, &scores)?;
        let weights = match mode.pass {
            Pass::Inference => candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?,
            Pass::Differentiable => candle_nn::ops::softmax(&scores, D::Minus1)?,
        };
        let attended = weights
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch, sequence, self.heads * self.head_dim))?
            .to_dtype(dtype)?;
        self.output_conv
            .forward(&self.output.forward(&attended)?, layer, cache)
    }
}
