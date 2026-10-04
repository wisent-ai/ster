//! Qwen3-Next's gated delta-rule linear attention (`Qwen3NextGatedDeltaNet`),
//! in place of attention on its linear-attention layers.
//!
//! `in_proj_qkvz` yields, per key head, its query and key and the values and
//! output gates of the value heads it serves; `in_proj_ba` yields each value
//! head's write strength `b` and decay input `a`. Query, key and value pass a
//! causal depthwise convolution and SiLU. Each value head keeps a `[key_dim,
//! value_dim]` state that decays by `exp(g)`, `g = -exp(A_log) ·
//! softplus(a + dt_bias)`, and is corrected toward the value by the delta
//! rule with strength `sigmoid(b)`; the query reads it. The read-out is RMS
//! normalised per head, scaled by `norm`, gated by SiLU of `z`, and
//! projected back by `out_proj`.
//!
//! The decode state is the convolution's last `kernel - 1` inputs and the
//! per-head states, kept in the cache's recurrent-state slot for the layer.

use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{Cache, DeltaRuleSpec};

use super::state_space::{causal_convolution, softplus};

/// The epsilon Transformers' `l2norm` adds before normalising each query
/// and key (`modeling_qwen3_next.py`, `l2norm(..., eps=1e-6)`).
const L2_NORM_EPS: f64 = 1e-6;

#[derive(Debug, Clone)]
pub(in crate::model::layer) struct DeltaRule {
    query_key_value_gate: Linear,
    strength_decay: Linear,
    /// The depthwise convolution over query, key and value, `[channels,
    /// kernel]`.
    convolution: Tensor,
    /// `dt_bias`, `[value_heads]`, in F32.
    step_bias: Tensor,
    /// `-exp(A_log)`, `[value_heads]`, in F32.
    decay: Tensor,
    /// The per-head norm's scale, `[value_dim]`.
    norm: Tensor,
    eps: f64,
    output: Linear,
    spec: DeltaRuleSpec,
}

impl DeltaRule {
    /// `builder` is the mixer's (`model.layers.{i}.linear_attn`).
    pub(in crate::model::layer) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        eps: f64,
        spec: DeltaRuleSpec,
    ) -> candle_core::Result<Self> {
        let DeltaRuleSpec { key_heads, value_heads, key_dim, value_dim, kernel, .. } = spec;
        if key_heads == 0 || value_heads % key_heads != 0 {
            candle_core::bail!(
                "{value_heads} value heads cannot be shared evenly among {key_heads} key heads"
            );
        }
        let keys = key_heads * key_dim;
        let values = value_heads * value_dim;
        let channels = 2 * keys + values;
        let f32_vector = |name: &str| -> candle_core::Result<Tensor> {
            builder.get(value_heads, name)?.to_dtype(DType::F32)
        };
        Ok(Self {
            query_key_value_gate: linear_no_bias(hidden, 2 * keys + 2 * values, builder.pp("in_proj_qkvz"))?,
            strength_decay: linear_no_bias(hidden, 2 * value_heads, builder.pp("in_proj_ba"))?,
            convolution: builder
                .pp("conv1d")
                .get((channels, 1, kernel), "weight")?
                .reshape((channels, kernel))?,
            step_bias: f32_vector("dt_bias")?,
            decay: f32_vector("A_log")?.exp()?.neg()?,
            norm: builder.pp("norm").get(value_dim, "weight")?,
            eps,
            output: linear_no_bias(values, hidden, builder.pp("out_proj"))?,
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
        let DeltaRuleSpec { key_heads, value_heads, key_dim, value_dim, kernel, .. } = self.spec;
        let shared = value_heads / key_heads;
        let dtype = hidden.dtype();
        let device = hidden.device();

        // Per key head: its query, its key, then the values and the output
        // gates of the `shared` value heads it serves.
        let per_head = 2 * key_dim + 2 * shared * value_dim;
        let projected = self
            .query_key_value_gate
            .forward(hidden)?
            .reshape((batch, sequence, key_heads, per_head))?;
        let part = |start: usize, width: usize| projected.narrow(3, start, width);
        let query = part(0, key_dim)?.reshape((batch, sequence, key_heads * key_dim))?;
        let key = part(key_dim, key_dim)?.reshape((batch, sequence, key_heads * key_dim))?;
        let value =
            part(2 * key_dim, shared * value_dim)?.reshape((batch, sequence, value_heads * value_dim))?;
        let gate = part(2 * key_dim + shared * value_dim, shared * value_dim)?
            .reshape((batch, sequence, value_heads, value_dim))?;
        let strength_decay = self
            .strength_decay
            .forward(hidden)?
            .reshape((batch, sequence, key_heads, 2 * shared))?;
        let strength = strength_decay
            .narrow(3, 0, shared)?
            .reshape((batch, sequence, value_heads))?
            .to_dtype(DType::F32)?;
        let decay_input = strength_decay
            .narrow(3, shared, shared)?
            .reshape((batch, sequence, value_heads))?
            .to_dtype(DType::F32)?;

        // The causal depthwise convolution and SiLU over query, key and
        // value, continued from the inputs the previous call ended on.
        let mixed = Tensor::cat(&[&query, &key, &value], 2)?;
        let saved = if cache.use_kv_cache { cache.states[layer].clone() } else { None };
        let (history, state) = match saved {
            Some((history, state)) => (Some(history), state),
            None => (
                None,
                Tensor::zeros((batch, value_heads, key_dim, value_dim), DType::F32, device)?,
            ),
        };
        let (convolved, next_history) =
            causal_convolution(&mixed.transpose(1, 2)?, history, &self.convolution, None, kernel)?;
        let mixed = candle_nn::ops::silu(&convolved)?.transpose(1, 2)?.contiguous()?;
        let keys = key_heads * key_dim;
        // Each key head's query and key serve its `shared` value heads; both
        // are L2-normalised and the query scaled by `1 / sqrt(key_dim)`.
        let by_value_head = |flat: Tensor| -> candle_core::Result<Tensor> {
            flat.reshape((batch, sequence, key_heads, 1, key_dim))?
                .broadcast_as((batch, sequence, key_heads, shared, key_dim))?
                .reshape((batch, sequence, value_heads, key_dim))
        };
        let l2 = |vectors: Tensor| -> candle_core::Result<Tensor> {
            let length = (vectors.sqr()?.sum_keepdim(D::Minus1)? + L2_NORM_EPS)?.sqrt()?;
            vectors.broadcast_div(&length)
        };
        let query = (l2(by_value_head(mixed.narrow(2, 0, keys)?)?)? / (key_dim as f64).sqrt())?;
        let key = l2(by_value_head(mixed.narrow(2, keys, keys)?)?)?;
        let value = mixed
            .narrow(2, 2 * keys, value_heads * value_dim)?
            .reshape((batch, sequence, value_heads, value_dim))?;

        // Write strength `sigmoid(b)`, composed so it has a backward pass,
        // and log-decay `g = -exp(A_log) · softplus(a + dt_bias)`.
        let strength = (strength.neg()?.exp()? + 1.0)?.recip()?;
        let log_decay = softplus(
            &decay_input.broadcast_add(&self.step_bias.reshape((1, 1, value_heads))?)?,
        )?
        .broadcast_mul(&self.decay.reshape((1, 1, value_heads))?)?;

        let mut state = state;
        let mut outputs = Vec::with_capacity(sequence);
        for position in 0..sequence {
            let query_t = query.i((.., position, .., ..))?;
            let key_t = key.i((.., position, .., ..))?;
            let value_t = value.i((.., position, .., ..))?;
            let decay_t = log_decay
                .i((.., position, ..))?
                .exp()?
                .reshape((batch, value_heads, 1, 1))?;
            let strength_t = strength.i((.., position, ..))?.reshape((batch, value_heads, 1))?;
            state = state.broadcast_mul(&decay_t)?;
            // What the state already recalls for this key, and the
            // delta-rule correction toward the value.
            let recalled = state.broadcast_mul(&key_t.unsqueeze(3)?)?.sum(2)?;
            let correction = (value_t - recalled)?.broadcast_mul(&strength_t)?;
            state = (state + key_t.unsqueeze(3)?.broadcast_mul(&correction.unsqueeze(2)?)?)?;
            outputs.push(state.broadcast_mul(&query_t.unsqueeze(3)?)?.sum(2)?);
        }
        let read = Tensor::stack(&outputs, 1)?;
        if cache.use_kv_cache {
            cache.states[layer] = Some((next_history, state));
        }

        // RMS norm per value head and the stored scale, then SiLU of the
        // gate, as Transformers' `Qwen3NextRMSNormGated` orders them.
        let square = (read.sqr()?.sum_keepdim(D::Minus1)? / value_dim as f64)?;
        let normed = read
            .broadcast_div(&(square + self.eps)?.sqrt()?)?
            .to_dtype(dtype)?
            .broadcast_mul(&self.norm)?
            .to_dtype(DType::F32)?;
        let gated = (normed * candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?
            .to_dtype(dtype)?
            .reshape((batch, sequence, value_heads * value_dim))?;
        self.output.forward(&gated)
    }
}
