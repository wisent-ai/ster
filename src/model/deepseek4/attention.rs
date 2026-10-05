//! DeepSeek-V4's attention (`DeepseekV4Attention`): queries from a latent
//! (`q_a_proj`, `q_a_norm`, `q_b_proj`) RMS-normed per head without a
//! scale; one key-value head (`kv_proj`, `kv_norm`) that is both key and
//! value; the trailing `qk_rope_head_dim` of every head rotated pair by pair;
//! a learned sink logit per head (`sinks`); every key within
//! `sliding_window`, and on the compressed layers the compressor's entries
//! closed before the query — on the sparse layers only those its indexer
//! keeps. The output's rotated slice is rotated back at the query's position
//! before `o_a_proj` projects each of `o_groups` groups of heads apart and
//! `o_b_proj` mixes them.

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::super::{
    Cache, CompressedSpec, Mode, NormKind, Pass,
    attention::apply_rotary,
    cache::{RotaryTable, base_frequencies, yarn_frequencies},
    layer::norm::{Norm, NormSpec},
};
use super::compressor::{Compressor, Indexer, closed_entries};

/// Rotates the trailing `2 · cos.dim(1)` channels of `input` `[batch,
/// heads, sequence, head_dim]` pair by pair; the rest pass through.
pub(super) fn rotate_trailing(input: &Tensor, cos: &Tensor, sin: &Tensor, pass: Pass) -> candle_core::Result<Tensor> {
    let head_dim = input.dim(3)?;
    let rotary = 2 * cos.dim(1)?;
    let dtype = input.dtype();
    let rotated = apply_rotary(&input.narrow(3, head_dim - rotary, rotary)?, cos, sin, dtype, pass, true)?;
    if rotary == head_dim {
        return Ok(rotated);
    }
    Tensor::cat(&[&input.narrow(3, 0, head_dim - rotary)?, &rotated], 3)
}

/// How a layer's attention reaches past its window.
#[derive(Debug, Clone)]
enum Reach {
    Window,
    Heavy(Compressor),
    Sparse(Compressor, Box<Indexer>),
}

#[derive(Debug, Clone)]
pub(in crate::model) struct CompressedAttention {
    query_down: Linear,
    query_norm: Norm,
    query_up: Linear,
    key_value: Linear,
    key_value_norm: Norm,
    output_down: Tensor,
    output_up: Linear,
    /// `[1, heads, 1, 1]` in F32.
    sinks: Tensor,
    reach: Reach,
    rotation: RotaryTable,
    heads: usize,
    head_dim: usize,
    groups: usize,
    window: usize,
    eps: f64,
}

impl CompressedAttention {
    /// `builder` is the layer's.
    pub(in crate::model) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: &CompressedSpec,
        eps: f64,
        layer: usize,
    ) -> candle_core::Result<Self> {
        let block = builder.pp("self_attn");
        let at = |set: u128| set >> layer & 1 == 1;
        let rms = NormSpec { kind: NormKind::Rms, eps, offset: false, groups: 1 };
        let width = spec.heads * spec.head_dim;
        let (reach, theta) = if at(spec.sparse_layers) {
            let compressor = block.pp("compressor");
            (
                Reach::Sparse(
                    Compressor::load(&compressor, hidden, spec.head_dim, spec.sparse_rate, true, eps, 0)?,
                    Box::new(Indexer::load(
                        &compressor.pp("indexer"),
                        hidden,
                        spec.query_rank,
                        (spec.index_heads, spec.index_head_dim, spec.index_top_k, spec.sparse_rate),
                        eps,
                    )?),
                ),
                spec.compress_theta,
            )
        } else if at(spec.heavy_layers) {
            let compressor = block.pp("compressor");
            (
                Reach::Heavy(Compressor::load(&compressor, hidden, spec.head_dim, spec.heavy_rate, false, eps, 0)?),
                spec.compress_theta,
            )
        } else {
            (Reach::Window, spec.main_theta)
        };
        let base = base_frequencies(spec.rotary_dim, theta);
        let frequencies = match (&reach, spec.compress_yarn) {
            (Reach::Window, _) | (_, None) => base,
            (_, Some(yarn)) => yarn_frequencies(
                &base,
                spec.rotary_dim,
                theta,
                yarn.factor,
                yarn.original,
                (yarn.beta_fast, yarn.beta_slow),
                yarn.truncate,
            ),
        };
        let groups = spec.output_groups;
        Ok(Self {
            query_down: linear_no_bias(hidden, spec.query_rank, block.pp("q_a_proj"))?,
            query_norm: rms.load(spec.query_rank, block.pp("q_a_norm"))?,
            query_up: linear_no_bias(spec.query_rank, width, block.pp("q_b_proj"))?,
            key_value: linear_no_bias(hidden, spec.head_dim, block.pp("kv_proj"))?,
            key_value_norm: rms.load(spec.head_dim, block.pp("kv_norm"))?,
            output_down: block
                .pp("o_a_proj")
                .get((groups * spec.output_rank, width / groups), "weight")?
                .reshape((groups, spec.output_rank, width / groups))?,
            output_up: linear_no_bias(groups * spec.output_rank, hidden, block.pp("o_b_proj"))?,
            sinks: block.get(spec.heads, "sinks")?.to_dtype(DType::F32)?.reshape((1, spec.heads, 1, 1))?,
            reach,
            rotation: RotaryTable::new(frequencies, 1.0, builder.device())?,
            heads: spec.heads,
            head_dim: spec.head_dim,
            groups,
            window: spec.window,
            eps,
        })
    }

    /// The window this layer's keys are read through.
    pub(in crate::model) fn window(&self) -> Option<usize> {
        Some(self.window)
    }

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
        let device = normed.device();
        let (cos, sin) = self.rotation.angles(index_pos, sequence)?;
        let latent = self.query_norm.forward(&self.query_down.forward(normed)?, mode.pass)?;
        let query = self.query_up.forward(&latent)?.reshape((batch, sequence, self.heads, self.head_dim))?;
        let variance = query.to_dtype(DType::F32)?.sqr()?.mean_keepdim(D::Minus1)?;
        let query = query.to_dtype(DType::F32)?.broadcast_div(&(variance + self.eps)?.sqrt()?)?.to_dtype(dtype)?;
        let query = rotate_trailing(&query.transpose(1, 2)?.contiguous()?, &cos, &sin, mode.pass)?;
        let key = self
            .key_value_norm
            .forward(&self.key_value.forward(normed)?, mode.pass)?
            .unsqueeze(1)?;
        let mut key = rotate_trailing(&key, &cos, &sin, mode.pass)?;
        if cache.use_kv_cache {
            if let Some((cached, _)) = &cache.kvs[layer] {
                key = Tensor::cat(&[cached, &key], 2)?.contiguous()?;
            }
            cache.kvs[layer] = Some((key.clone(), key.clone()));
        }
        let total = key.dim(2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let negative = |shape: &candle_core::Shape| Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as(shape.clone());
        let scores = (query.broadcast_matmul(&key.transpose(2, 3)?.contiguous()?)? * scale)?;
        let hidden = match mask {
            Some(mask) => mask.broadcast_as(scores.shape())?.to_dtype(DType::U8)?,
            None => cache.mask(sequence, index_pos, Some(self.window))?.broadcast_as(scores.shape())?,
        };
        let scores = hidden.where_cond(&negative(scores.shape())?, &scores)?;
        // The compressed entries closed before each query, `[batch, 1,
        // entries, head_dim]`, and which of them each query may not see.
        let compressed = match &self.reach {
            Reach::Window => None,
            Reach::Heavy(compressor) => {
                let entries = compressor.entries(normed, layer, cache, &self.rotation, mode.pass)?;
                let count = entries.dim(1)?;
                let hidden = closed_entries(index_pos, sequence, count, compressor.rate(), device)?;
                Some((entries, hidden.broadcast_as((batch, sequence, count))?.contiguous()?))
            }
            Reach::Sparse(compressor, indexer) => {
                let entries = compressor.entries(normed, layer, cache, &self.rotation, mode.pass)?;
                let hidden = indexer.hidden_entries(normed, &latent, index_pos, layer, cache, &self.rotation, mode.pass)?;
                if hidden.dim(2)? != entries.dim(1)? {
                    candle_core::bail!(
                        "layer {layer}'s indexer ranked {} entries and its compressor holds {}",
                        hidden.dim(2)?,
                        entries.dim(1)?
                    );
                }
                Some((entries, hidden))
            }
        };
        let compressed = match compressed {
            Some((entries, hidden)) if entries.dim(1)? > 0 => {
                let entries = entries.to_dtype(DType::F32)?.unsqueeze(1)?;
                let scores = (query.broadcast_matmul(&entries.transpose(2, 3)?.contiguous()?)? * scale)?;
                let hidden = hidden.unsqueeze(1)?.broadcast_as(scores.shape())?;
                Some((entries, hidden.where_cond(&negative(scores.shape())?, &scores)?))
            }
            _ => None,
        };
        let mut logits = vec![scores];
        if let Some((_, scores)) = &compressed {
            logits.push(scores.clone());
        }
        logits.push(self.sinks.broadcast_as((batch, self.heads, sequence, 1))?.contiguous()?);
        let logits = Tensor::cat(&logits, 3)?;
        let probabilities = match mode.pass {
            Pass::Inference => candle_nn::ops::softmax_last_dim(&logits.contiguous()?)?,
            Pass::Differentiable => candle_nn::ops::softmax(&logits, D::Minus1)?,
        };
        let mut output = probabilities.narrow(3, 0, total)?.broadcast_matmul(&key)?;
        if let Some((entries, _)) = &compressed {
            let count = entries.dim(2)?;
            output = (output + probabilities.narrow(3, total, count)?.broadcast_matmul(entries)?)?;
        }
        // Undo the rotation the values carry, at the query's position.
        let output = rotate_trailing(&output, &cos, &sin.neg()?, mode.pass)?;
        let grouped = output
            .transpose(1, 2)?
            .reshape((batch * sequence, self.groups, self.heads * self.head_dim / self.groups))?
            .transpose(0, 1)?
            .contiguous()?;
        let projected = grouped
            .matmul(&self.output_down.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?)?
            .transpose(0, 1)?
            .reshape((batch, sequence, ()))?
            .to_dtype(dtype)?;
        self.output_up.forward(&projected)
    }
}
