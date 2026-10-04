//! Where each delta-rule form reads its query, key, value, output gate,
//! write strength and decay input from.
//!
//! * **Qwen3-Next** — `in_proj_qkvz` yields, per key head, its query and key
//!   and the values and output gates of the value heads it serves;
//!   `in_proj_ba` yields each value head's `b` and decay input `a`, one per
//!   value head; one `conv1d` covers query, key and value.
//! * **Kimi** — separate `q_proj`, `k_proj`, `v_proj` with their own
//!   `q_conv1d`, `k_conv1d`, `v_conv1d`; `b_proj` yields `b`; the decay input
//!   is `f_b(f_a(x))`, one per key channel; the gate is `g_b(g_a(x))`.
//! * **OLMo Hybrid** — separate `q_proj`, `k_proj`, `v_proj` under one
//!   `conv1d` (or OLMo-core's `q_conv1d`, `k_conv1d`, `v_conv1d`); `b_proj`
//!   and `a_proj` yield `b` and the decay input, one per value head;
//!   `g_proj` the gate.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{DeltaRuleForm, DeltaRuleSpec};

#[derive(Debug, Clone)]
pub(super) enum Inputs {
    Qwen3Next {
        query_key_value_gate: Linear,
        strength_decay: Linear,
    },
    Kimi {
        query: Linear,
        key: Linear,
        value: Linear,
        strength: Linear,
        forget_down: Linear,
        forget_up: Linear,
        gate_down: Linear,
        gate_up: Linear,
    },
    OlmoHybrid {
        query: Linear,
        key: Linear,
        value: Linear,
        strength: Linear,
        decay: Linear,
        gate: Linear,
    },
}

/// What one call's inputs come to before the convolution.
pub(super) struct Prepared {
    /// Query, key and value side by side, `[batch, sequence, channels]`.
    pub mixed: Tensor,
    /// The output gate's pre-activation, `[batch, sequence, value_heads,
    /// value_dim]`.
    pub gate: Tensor,
    /// `b`, `[batch, sequence, value_heads]`, in F32.
    pub strength: Tensor,
    /// The decay's softplus input before `dt_bias`, `[batch, sequence,
    /// heads, 1 or key_dim]`, in F32.
    pub decay_input: Tensor,
}

impl Inputs {
    /// The form's projections, its convolution taps `[channels, kernel]`,
    /// and its `dt_bias` shaped to add to the decay input, in F32.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: DeltaRuleSpec,
    ) -> candle_core::Result<(Self, Tensor, Tensor)> {
        let DeltaRuleSpec { key_heads, value_heads, key_dim, value_dim, kernel, form, .. } = spec;
        let keys = key_heads * key_dim;
        let values = value_heads * value_dim;
        let taps = |name: &str, width: usize| -> candle_core::Result<Tensor> {
            builder.pp(name).get((width, 1, kernel), "weight")?.reshape((width, kernel))
        };
        Ok(match form {
            DeltaRuleForm::Qwen3Next => (
                Self::Qwen3Next {
                    query_key_value_gate: linear_no_bias(hidden, 2 * keys + 2 * values, builder.pp("in_proj_qkvz"))?,
                    strength_decay: linear_no_bias(hidden, 2 * value_heads, builder.pp("in_proj_ba"))?,
                },
                taps("conv1d", 2 * keys + values)?,
                builder.get(value_heads, "dt_bias")?.to_dtype(DType::F32)?.reshape((1, 1, value_heads, 1))?,
            ),
            DeltaRuleForm::Kimi => (
                Self::Kimi {
                    query: linear_no_bias(hidden, keys, builder.pp("q_proj"))?,
                    key: linear_no_bias(hidden, keys, builder.pp("k_proj"))?,
                    value: linear_no_bias(hidden, values, builder.pp("v_proj"))?,
                    strength: linear_no_bias(hidden, value_heads, builder.pp("b_proj"))?,
                    forget_down: linear_no_bias(hidden, key_dim, builder.pp("f_a_proj"))?,
                    forget_up: linear_no_bias(key_dim, keys, builder.pp("f_b_proj"))?,
                    gate_down: linear_no_bias(hidden, value_dim, builder.pp("g_a_proj"))?,
                    gate_up: linear_no_bias(value_dim, values, builder.pp("g_b_proj"))?,
                },
                Tensor::cat(&[&taps("q_conv1d", keys)?, &taps("k_conv1d", keys)?, &taps("v_conv1d", values)?], 0)?,
                builder.get(keys, "dt_bias")?.to_dtype(DType::F32)?.reshape((1, 1, key_heads, key_dim))?,
            ),
            DeltaRuleForm::OlmoHybrid => (
                Self::OlmoHybrid {
                    query: linear_no_bias(hidden, keys, builder.pp("q_proj"))?,
                    key: linear_no_bias(hidden, keys, builder.pp("k_proj"))?,
                    value: linear_no_bias(hidden, values, builder.pp("v_proj"))?,
                    strength: linear_no_bias(hidden, value_heads, builder.pp("b_proj"))?,
                    decay: linear_no_bias(hidden, value_heads, builder.pp("a_proj"))?,
                    gate: linear_no_bias(hidden, values, builder.pp("g_proj"))?,
                },
                // OLMo-core's checkpoints keep one convolution per input
                // (`q_conv1d`, `k_conv1d`, `v_conv1d`), which Transformers
                // joins into its single `conv1d` in that order.
                if builder.contains_tensor("conv1d.weight") {
                    taps("conv1d", 2 * keys + values)?
                } else {
                    Tensor::cat(&[&taps("q_conv1d", keys)?, &taps("k_conv1d", keys)?, &taps("v_conv1d", values)?], 0)?
                },
                builder.get(value_heads, "dt_bias")?.to_dtype(DType::F32)?.reshape((1, 1, value_heads, 1))?,
            ),
        })
    }

    pub(super) fn prepare(&self, hidden: &Tensor, spec: DeltaRuleSpec) -> candle_core::Result<Prepared> {
        let (batch, sequence, _) = hidden.dims3()?;
        let DeltaRuleSpec { key_heads, value_heads, key_dim, value_dim, .. } = spec;
        let shared = value_heads / key_heads;
        match self {
            Self::Qwen3Next { query_key_value_gate, strength_decay } => {
                let per_head = 2 * key_dim + 2 * shared * value_dim;
                let projected = query_key_value_gate
                    .forward(hidden)?
                    .reshape((batch, sequence, key_heads, per_head))?;
                let part = |start: usize, width: usize, flat: usize| -> candle_core::Result<Tensor> {
                    projected.narrow(3, start, width)?.reshape((batch, sequence, flat))
                };
                let query = part(0, key_dim, key_heads * key_dim)?;
                let key = part(key_dim, key_dim, key_heads * key_dim)?;
                let value = part(2 * key_dim, shared * value_dim, value_heads * value_dim)?;
                let gate = projected
                    .narrow(3, 2 * key_dim + shared * value_dim, shared * value_dim)?
                    .reshape((batch, sequence, value_heads, value_dim))?;
                let strength_decay = strength_decay
                    .forward(hidden)?
                    .reshape((batch, sequence, key_heads, 2 * shared))?;
                Ok(Prepared {
                    mixed: Tensor::cat(&[&query, &key, &value], 2)?,
                    gate,
                    strength: strength_decay
                        .narrow(3, 0, shared)?
                        .reshape((batch, sequence, value_heads))?
                        .to_dtype(DType::F32)?,
                    decay_input: strength_decay
                        .narrow(3, shared, shared)?
                        .reshape((batch, sequence, value_heads, 1))?
                        .to_dtype(DType::F32)?,
                })
            }
            Self::Kimi { query, key, value, strength, forget_down, forget_up, gate_down, gate_up } => Ok(Prepared {
                mixed: Tensor::cat(&[&query.forward(hidden)?, &key.forward(hidden)?, &value.forward(hidden)?], 2)?,
                gate: gate_up
                    .forward(&gate_down.forward(hidden)?)?
                    .reshape((batch, sequence, value_heads, value_dim))?,
                strength: strength.forward(hidden)?.to_dtype(DType::F32)?,
                decay_input: forget_up
                    .forward(&forget_down.forward(hidden)?)?
                    .reshape((batch, sequence, key_heads, key_dim))?
                    .to_dtype(DType::F32)?,
            }),
            Self::OlmoHybrid { query, key, value, strength, decay, gate } => Ok(Prepared {
                mixed: Tensor::cat(&[&query.forward(hidden)?, &key.forward(hidden)?, &value.forward(hidden)?], 2)?,
                gate: gate.forward(hidden)?.reshape((batch, sequence, value_heads, value_dim))?,
                strength: strength.forward(hidden)?.to_dtype(DType::F32)?,
                decay_input: decay
                    .forward(hidden)?
                    .reshape((batch, sequence, value_heads, 1))?
                    .to_dtype(DType::F32)?,
            }),
        }
    }
}
