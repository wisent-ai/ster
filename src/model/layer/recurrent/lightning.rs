//! MiniMax-Text-01's lightning attention (`MiniMaxLightningAttention`), in
//! place of attention on its linear-attention layers.
//!
//! `qkv_proj` yields each head's query, key and value, passed through SiLU.
//! Each head keeps a `[head_dim, head_dim]` state that decays by
//! `exp(-slope)` per position and gains `kᵀv`; the query reads it. The
//! read-out is RMS normalised over all heads together, scaled by `norm`,
//! multiplied by the sigmoid of `output_gate(x)`, and projected back by
//! `out_proj`.
//!
//! Each head's slope is `base^(head + 1)` with `base = 2^(-8 / heads)`,
//! scaled by `1 - layer / (layers - 1 + 1e-5) + 1e-5`, as Transformers'
//! `get_slope_rate` computes it. The decode state is the per-head state,
//! kept in the cache's recurrent-state slot for the layer.

use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{Cache, LightningSpec};

/// The span of the slope base's exponent in `get_slope_rate`
/// (`modeling_minimax.py`: `1 / (2 ** (8 / num_attention_heads))`).
const SLOPE_EXPONENT_SPAN: f64 = 8.0;

/// The offset `get_slope_rate` adds twice so the last layer's slope stays
/// above zero (`modeling_minimax.py`: `+ 1e-5`).
const SLOPE_FACTOR_OFFSET: f64 = 1e-5;

/// The epsilon of the lightning attention's output norm, `MiniMaxRMSNorm`'s
/// default, which the layer does not override (`modeling_minimax.py`).
const NORM_EPS: f64 = 1e-6;

#[derive(Debug, Clone)]
pub(in crate::model::layer) struct Lightning {
    query_key_value: Linear,
    gate: Linear,
    /// The output norm's scale, `[heads · head_dim]`.
    norm: Tensor,
    output: Linear,
    /// `exp(-slope)` per head, `[1, heads, 1, 1]`, in F32.
    decay: Tensor,
    spec: LightningSpec,
}

impl Lightning {
    /// `builder` is the mixer's (`model.layers.{i}.self_attn`); `layer` and
    /// `layers` set the slopes.
    pub(in crate::model::layer) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        layer: usize,
        layers: usize,
        spec: LightningSpec,
    ) -> candle_core::Result<Self> {
        let LightningSpec { heads, head_dim, .. } = spec;
        let width = heads * head_dim;
        let base = 2f64.powf(-SLOPE_EXPONENT_SPAN / heads as f64);
        let factor = 1.0 - layer as f64 / (layers as f64 - 1.0 + SLOPE_FACTOR_OFFSET) + SLOPE_FACTOR_OFFSET;
        let decay: Vec<f32> = (1..=heads)
            .map(|head| (-(base.powi(head as i32) * factor)).exp() as f32)
            .collect();
        Ok(Self {
            query_key_value: linear_no_bias(hidden, 3 * width, builder.pp("qkv_proj"))?,
            gate: linear_no_bias(hidden, width, builder.pp("output_gate"))?,
            norm: builder.pp("norm").get(width, "weight")?,
            output: linear_no_bias(width, hidden, builder.pp("out_proj"))?,
            decay: Tensor::from_vec(decay, (1, heads, 1, 1), builder.device())?,
            spec,
        })
    }

    /// Mixes `hidden` `[batch, sequence, width]`, continuing from the
    /// layer's saved state when the cache keeps one. Every op has a backward
    /// pass, so the same code serves inference and training.
    pub(in crate::model::layer) fn forward(
        &self,
        hidden: &Tensor,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let LightningSpec { heads, head_dim, .. } = self.spec;
        let dtype = hidden.dtype();
        let projected = candle_nn::ops::silu(&self.query_key_value.forward(hidden)?)?
            .reshape((batch, sequence, heads, 3 * head_dim))?
            .to_dtype(DType::F32)?;
        let query = projected.narrow(3, 0, head_dim)?;
        let key = projected.narrow(3, head_dim, head_dim)?;
        let value = projected.narrow(3, 2 * head_dim, head_dim)?;

        let saved = if cache.use_kv_cache { cache.states[layer].clone() } else { None };
        let mut state = match saved {
            Some((_, state)) => state,
            None => Tensor::zeros((batch, heads, head_dim, head_dim), DType::F32, hidden.device())?,
        };
        let mut outputs = Vec::with_capacity(sequence);
        for position in 0..sequence {
            let key_t = key.i((.., position, .., ..))?;
            let value_t = value.i((.., position, .., ..))?;
            let query_t = query.i((.., position, .., ..))?;
            state = (state.broadcast_mul(&self.decay)?
                + key_t.unsqueeze(3)?.broadcast_mul(&value_t.unsqueeze(2)?)?)?;
            outputs.push(state.broadcast_mul(&query_t.unsqueeze(3)?)?.sum(2)?);
        }
        let read = Tensor::stack(&outputs, 1)?.reshape((batch, sequence, heads * head_dim))?;
        if cache.use_kv_cache {
            // The slot holds a pair; lightning attention has no convolution
            // history, so both are the state.
            cache.states[layer] = Some((state.clone(), state));
        }

        // RMS norm over every head together, the stored scale, then the
        // sigmoid output gate, composed so it has a backward pass.
        let square = read.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = read
            .broadcast_div(&(square + NORM_EPS)?.sqrt()?)?
            .broadcast_mul(&self.norm.to_dtype(DType::F32)?)?;
        let gate = self.gate.forward(hidden)?.to_dtype(DType::F32)?;
        let gate = (gate.neg()?.exp()? + 1.0)?.recip()?;
        self.output.forward(&(normed * gate)?.to_dtype(dtype)?)
    }
}
