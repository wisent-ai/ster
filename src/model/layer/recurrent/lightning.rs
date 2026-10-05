//! Lightning attention, in place of attention on a family's linear-attention
//! layers: MiniMax-Text-01's (`MiniMaxLightningAttention`) and Ling 2.5's
//! (`BailingMoELinearAttention`).
//!
//! Each head's query, key and value come from one projection (through SiLU
//! for MiniMax, and for Ling under `linear_silu`). Ling normalises each
//! head's query and key, rotates them, and scales the query by
//! `head_dim^-0.5`. Each head keeps a `[head_dim, head_dim]` state that
//! decays by `exp(-slope)` per position and gains `kᵀv`; the query reads it.
//! The read-out is RMS normalised — over all heads together for MiniMax,
//! over `groups` equal groups for Ling — scaled by the stored norm,
//! multiplied by the sigmoid of a gate projection, and projected back.
//!
//! Each head's slope is `base^(head + 1)` with `base = 2^(-8 / heads)`,
//! scaled by `1 - layer / (layers - 1 + 1e-5) + 1e-5` (MiniMax's
//! `get_slope_rate`) or `1 - layer / (layers - 1) + 1e-5` (vLLM's
//! `BailingMoELinearAttention`). The decode state is the per-head state,
//! kept in the cache's recurrent-state slot for the layer.

use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{Cache, LightningForm, LightningSpec, Mode, attention::apply_rotary};

use super::super::norm::{Norm, NormSpec};

/// The span of the slope base's exponent in `get_slope_rate`
/// (`modeling_minimax.py`: `1 / (2 ** (8 / num_attention_heads))`).
const SLOPE_EXPONENT_SPAN: f64 = 8.0;

/// The offset `get_slope_rate` adds so the last layer's slope stays above
/// zero (`modeling_minimax.py` and vLLM's `bailing_linear_attn.py`:
/// `+ 1e-5`).
const SLOPE_FACTOR_OFFSET: f64 = 1e-5;

/// The epsilon of MiniMax's lightning output norm, `MiniMaxRMSNorm`'s
/// default, which the layer does not override (`modeling_minimax.py`).
const MINIMAX_NORM_EPS: f64 = 1e-6;

#[derive(Debug, Clone)]
pub(in crate::model::layer) struct Lightning {
    query_key_value: Linear,
    gate: Linear,
    /// The output norm's scale, `[heads · head_dim]`.
    norm: Tensor,
    norm_eps: f64,
    output: Linear,
    /// Ling's per-head query and key norms.
    query_key_norms: Option<(Norm, Norm)>,
    /// `exp(-slope)` per head, `[1, heads, 1, 1]`, in F32.
    decay: Tensor,
    spec: LightningSpec,
}

impl Lightning {
    /// `builder` is the mixer's (`model.layers.{i}.self_attn` or
    /// `.attention`); `layer` and `layers` set the slopes; `norms` and `eps`
    /// are the model's.
    pub(in crate::model::layer) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        layer: usize,
        layers: usize,
        spec: LightningSpec,
        norms: NormSpec,
    ) -> candle_core::Result<Self> {
        let LightningSpec { heads, head_dim, form, .. } = spec;
        let width = heads * head_dim;
        let base = 2f64.powf(-SLOPE_EXPONENT_SPAN / heads as f64);
        let span = match form {
            LightningForm::MiniMax => layers as f64 - 1.0 + SLOPE_FACTOR_OFFSET,
            LightningForm::Bailing { .. } => (layers as f64 - 1.0).max(1.0),
        };
        let factor = 1.0 - layer as f64 / span + SLOPE_FACTOR_OFFSET;
        let decay: Vec<f32> = (1..=heads)
            .map(|head| (-(base.powi(head as i32) * factor)).exp() as f32)
            .collect();
        let (projection, gate, norm, output) = match form {
            LightningForm::MiniMax => ("qkv_proj", "output_gate", "norm", "out_proj"),
            LightningForm::Bailing { .. } => ("query_key_value", "g_proj", "g_norm", "dense"),
        };
        let query_key_norms = match form {
            LightningForm::Bailing { qk_norm: true, .. } => Some((
                norms.load(head_dim, builder.pp("query_layernorm"))?,
                norms.load(head_dim, builder.pp("key_layernorm"))?,
            )),
            _ => None,
        };
        Ok(Self {
            query_key_value: linear_no_bias(hidden, 3 * width, builder.pp(projection))?,
            gate: linear_no_bias(hidden, width, builder.pp(gate))?,
            norm: builder.pp(norm).get(width, "weight")?,
            norm_eps: match form {
                LightningForm::MiniMax => MINIMAX_NORM_EPS,
                LightningForm::Bailing { .. } => norms.eps,
            },
            output: linear_no_bias(width, hidden, builder.pp(output))?,
            query_key_norms,
            decay: Tensor::from_vec(decay, (1, heads, 1, 1), builder.device())?,
            spec,
        })
    }

    /// Mixes `hidden` `[batch, sequence, width]` at positions from
    /// `index_pos`, continuing from the layer's saved state when the cache
    /// keeps one. Every op has a backward pass, so the same code serves
    /// inference and training.
    pub(in crate::model::layer) fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let LightningSpec { heads, head_dim, form, .. } = self.spec;
        let dtype = hidden.dtype();
        let projected = self.query_key_value.forward(hidden)?;
        // MiniMax lays each head's query, key and value side by side; Ling
        // stacks every query, then every key, then every value.
        let (query, key, value) = match form {
            LightningForm::MiniMax => {
                let projected = candle_nn::ops::silu(&projected)?
                    .reshape((batch, sequence, heads, 3 * head_dim))?
                    .to_dtype(DType::F32)?;
                (
                    projected.narrow(3, 0, head_dim)?,
                    projected.narrow(3, head_dim, head_dim)?,
                    projected.narrow(3, 2 * head_dim, head_dim)?,
                )
            }
            LightningForm::Bailing { silu, .. } => {
                let projected = if silu { candle_nn::ops::silu(&projected)? } else { projected };
                let width = heads * head_dim;
                let part = |start: usize| {
                    projected.narrow(2, start, width)?.contiguous()?.reshape((batch, sequence, heads, head_dim))
                };
                let (query, key, value) = (part(0)?, part(width)?, part(2 * width)?);
                // The per-head norms run at the weights' dtype, then
                // everything widens to F32.
                let (query, key) = match &self.query_key_norms {
                    Some((query_norm, key_norm)) => {
                        (query_norm.forward(&query, mode.pass)?, key_norm.forward(&key, mode.pass)?)
                    }
                    None => (query, key),
                };
                let value = value.to_dtype(DType::F32)?;
                // The global rotation by halves, over the rotated share of
                // each head.
                let (cos, sin) = cache.global.angles(index_pos, sequence)?;
                let rotate = |input: Tensor| -> candle_core::Result<Tensor> {
                    apply_rotary(&input.transpose(1, 2)?.contiguous()?, &cos, &sin, DType::F32, mode.pass, false)?
                        .transpose(1, 2)
                };
                let query = (rotate(query)? / (head_dim as f64).sqrt())?;
                (query, rotate(key)?, value)
            }
        };

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

        // RMS norm over every head together (MiniMax) or over each of Ling's
        // groups, the stored scale, then the sigmoid gate, composed so it has
        // a backward pass.
        let groups = match form {
            LightningForm::MiniMax => 1,
            LightningForm::Bailing { groups, .. } => groups.max(1),
        };
        let grouped = read.reshape((batch, sequence, groups, heads * head_dim / groups))?;
        let square = grouped.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = grouped
            .broadcast_div(&(square + self.norm_eps)?.sqrt()?)?
            .reshape((batch, sequence, heads * head_dim))?
            .broadcast_mul(&self.norm.to_dtype(DType::F32)?)?;
        let gate = self.gate.forward(hidden)?.to_dtype(DType::F32)?;
        let gate = (gate.neg()?.exp()? + 1.0)?.recip()?;
        self.output.forward(&(normed * gate)?.to_dtype(dtype)?)
    }
}
