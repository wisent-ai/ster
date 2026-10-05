//! Gated delta-rule linear attention, in place of attention on a family's
//! linear-attention layers: Qwen3-Next's Gated DeltaNet and Kimi Delta
//! Attention (see [`inputs`] for where each reads its inputs).
//!
//! Query, key and value pass a causal depthwise convolution and SiLU; query
//! and key are L2-normalised and the query scaled by one over the square
//! root of its width. Each value head keeps a `[key_dim, value_dim]` state
//! that decays by `exp(g)`, `g = -exp(A_log) · softplus(input + dt_bias)`
//! (one per value head, or one per key channel), and is corrected toward the
//! value by the delta rule with strength `sigmoid(b)`; the query reads it.
//! The read-out is RMS normalised per head, scaled by the stored norm, gated
//! (SiLU for Qwen3-Next, sigmoid for Kimi) and projected back.
//!
//! The decode state is the convolution's last `kernel - 1` inputs and the
//! per-head states, kept in the cache's recurrent-state slot for the layer.

mod inputs;

use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{Cache, DeltaRuleForm, DeltaRuleSpec};

use super::state_space::{causal_convolution, softplus};
use inputs::{Inputs, Prepared};

/// The epsilon Transformers' `l2norm` adds before normalising each query
/// and key (`modeling_qwen3_next.py` and `modeling_kimi_linear.py`,
/// `l2norm(..., eps=1e-6)`).
const L2_NORM_EPS: f64 = 1e-6;

/// The epsilon of OLMo Hybrid's gated output norm, which it fixes rather
/// than reading `rms_norm_eps` (`modeling_olmo_hybrid.py`,
/// `OlmoHybridRMSNormGated(self.head_v_dim, eps=1e-5)`).
const OLMO_HYBRID_NORM_EPS: f64 = 1e-5;

#[derive(Debug, Clone)]
pub(in crate::model::layer) struct DeltaRule {
    inputs: Inputs,
    /// The depthwise convolution over query, key and value, `[channels,
    /// kernel]`.
    convolution: Tensor,
    /// `dt_bias`, shaped to add to the decay input, in F32.
    step_bias: Tensor,
    /// `-exp(A_log)`, `[1, 1, heads, 1]`, in F32.
    decay: Tensor,
    /// The per-head norm's scale, `[value_dim]`.
    norm: Tensor,
    eps: f64,
    output: Linear,
    spec: DeltaRuleSpec,
}

impl DeltaRule {
    /// `builder` is the mixer's (`model.layers.{i}.linear_attn` or
    /// `model.layers.{i}.self_attn`).
    pub(in crate::model::layer) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        eps: f64,
        spec: DeltaRuleSpec,
    ) -> candle_core::Result<Self> {
        let DeltaRuleSpec { key_heads, value_heads, value_dim, form, .. } = spec;
        if key_heads == 0 || value_heads % key_heads != 0 {
            candle_core::bail!(
                "{value_heads} value heads cannot be shared evenly among {key_heads} key heads"
            );
        }
        let (inputs, convolution, step_bias) = Inputs::load(&builder, hidden, spec)?;
        let (norm, output) = match form {
            DeltaRuleForm::Qwen3Next | DeltaRuleForm::Qwen35 => ("norm", "out_proj"),
            DeltaRuleForm::Kimi | DeltaRuleForm::OlmoHybrid => ("o_norm", "o_proj"),
        };
        let eps = if form == DeltaRuleForm::OlmoHybrid { OLMO_HYBRID_NORM_EPS } else { eps };
        Ok(Self {
            inputs,
            convolution,
            step_bias,
            decay: builder
                .get_unchecked("A_log")?
                .flatten_all()?
                .to_dtype(DType::F32)?
                .exp()?
                .neg()?
                .reshape((1, 1, value_heads, 1))?,
            norm: builder.pp(norm).get(value_dim, "weight")?,
            eps,
            output: linear_no_bias(value_heads * value_dim, hidden, builder.pp(output))?,
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
        let DeltaRuleSpec { key_heads, value_heads, key_dim, value_dim, kernel, form, .. } = self.spec;
        let shared = value_heads / key_heads;
        let dtype = hidden.dtype();
        let device = hidden.device();
        let Prepared { mixed, gate, strength, decay_input } = self.inputs.prepare(hidden, self.spec)?;

        // The causal depthwise convolution and SiLU over query, key and
        // value, continued from the inputs the previous call ended on.
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
        // Each key head's query and key serve its `shared` value heads.
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
        // and log-decay `g`; Kimi's per-key-channel decay belongs to its key
        // head and is shared like the key.
        let strength = (strength.neg()?.exp()? + 1.0)?.recip()?;
        let strength = if self.spec.negative_eigenvalues { (strength * 2.0)? } else { strength };
        let log_decay =
            softplus(&decay_input.broadcast_add(&self.step_bias)?)?.broadcast_mul(&self.decay)?;
        let decay_width = log_decay.dim(3)?;

        let mut state = state;
        let mut outputs = Vec::with_capacity(sequence);
        for position in 0..sequence {
            let query_t = query.i((.., position, .., ..))?;
            let key_t = key.i((.., position, .., ..))?;
            let value_t = value.i((.., position, .., ..))?;
            let decay_t = log_decay
                .i((.., position, .., ..))?
                .exp()?
                .reshape((batch, value_heads, decay_width, 1))?;
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

        // RMS norm per value head and the stored scale, then the gate.
        let square = (read.sqr()?.sum_keepdim(D::Minus1)? / value_dim as f64)?;
        let normed = read
            .broadcast_div(&(square + self.eps)?.sqrt()?)?
            .broadcast_mul(&self.norm.to_dtype(DType::F32)?)?;
        let gate = gate.to_dtype(DType::F32)?;
        let gate = match form {
            DeltaRuleForm::Qwen3Next | DeltaRuleForm::Qwen35 | DeltaRuleForm::OlmoHybrid => candle_nn::ops::silu(&gate)?,
            DeltaRuleForm::Kimi => (gate.neg()?.exp()? + 1.0)?.recip()?,
        };
        let gated = (normed * gate)?
            .to_dtype(dtype)?
            .reshape((batch, sequence, value_heads * value_dim))?;
        self.output.forward(&gated)
    }
}
