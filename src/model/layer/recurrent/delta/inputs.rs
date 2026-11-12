//! Where each delta-rule form reads its query, key, value, output gate,
//! write strength and decay input from.
//!
//! * **Qwen3-Next** — `in_proj_qkvz` yields, per key head, its query and key
//!   and the values and output gates of the value heads it serves;
//!   `in_proj_ba` yields each value head's `b` and decay input `a`, one per
//!   value head; one `conv1d` covers query, key and value.
//! * **Kimi** — separate `q_proj`, `k_proj`, `v_proj` with their own
//!   `q_conv1d`, `k_conv1d`, `v_conv1d`; `b_proj` yields `b`; the decay input
//!   is `f_b(f_a(x))`, one per key channel; the gate is `g_b(g_a(x))`. Ling
//!   3.0's is Kimi's with full-rank `f_proj` and `g_proj`.
//! * **OLMo Hybrid** — separate `q_proj`, `k_proj`, `v_proj` under one
//!   `conv1d` (or OLMo-core's `q_conv1d`, `k_conv1d`, `v_conv1d`); `b_proj`
//!   and `a_proj` yield `b` and the decay input, one per value head;
//!   `g_proj` the gate.
//! * **Qwen3.5** — `in_proj_qkv` yields every query, then every key, then
//!   every value under one `conv1d`; `in_proj_b` and `in_proj_a` yield `b`
//!   and the decay input, one per value head; `in_proj_z` the gate.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module};

use crate::model::DeltaRuleSpec;

mod load;

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
        /// The decay input's projections, Kimi's `f_a_proj` then
        /// `f_b_proj`, or Ling's one `f_proj`.
        forget: (Option<Linear>, Linear),
        /// The output gate's, `g_a_proj` then `g_b_proj`, or `g_proj`.
        gate: (Option<Linear>, Linear),
    },
    OlmoHybrid {
        query: Linear,
        key: Linear,
        value: Linear,
        strength: Linear,
        decay: Linear,
        gate: Linear,
    },
    Qwen35 {
        query_key_value: Linear,
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
    pub(super) fn prepare(
        &self,
        hidden: &Tensor,
        spec: DeltaRuleSpec,
    ) -> candle_core::Result<Prepared> {
        let (batch, sequence, _) = hidden.dims3()?;
        let DeltaRuleSpec {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            ..
        } = spec;
        let shared = value_heads / key_heads;
        match self {
            Self::Qwen3Next {
                query_key_value_gate,
                strength_decay,
            } => {
                let per_head = 2 * key_dim + 2 * shared * value_dim;
                let projected = query_key_value_gate
                    .forward(hidden)?
                    .reshape((batch, sequence, key_heads, per_head))?;
                let part =
                    |start: usize, width: usize, flat: usize| -> candle_core::Result<Tensor> {
                        projected
                            .narrow(3, start, width)?
                            .reshape((batch, sequence, flat))
                    };
                let query = part(0, key_dim, key_heads * key_dim)?;
                let key = part(key_dim, key_dim, key_heads * key_dim)?;
                let value = part(2 * key_dim, shared * value_dim, value_heads * value_dim)?;
                let gate = projected
                    .narrow(3, 2 * key_dim + shared * value_dim, shared * value_dim)?
                    .reshape((batch, sequence, value_heads, value_dim))?;
                let strength_decay = strength_decay.forward(hidden)?.reshape((
                    batch,
                    sequence,
                    key_heads,
                    2 * shared,
                ))?;
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
            Self::Kimi {
                query,
                key,
                value,
                strength,
                forget,
                gate,
            } => {
                let through =
                    |(down, up): &(Option<Linear>, Linear)| -> candle_core::Result<Tensor> {
                        match down {
                            Some(down) => up.forward(&down.forward(hidden)?),
                            None => up.forward(hidden),
                        }
                    };
                Ok(Prepared {
                    mixed: Tensor::cat(
                        &[
                            &query.forward(hidden)?,
                            &key.forward(hidden)?,
                            &value.forward(hidden)?,
                        ],
                        2,
                    )?,
                    gate: through(gate)?.reshape((batch, sequence, value_heads, value_dim))?,
                    strength: strength.forward(hidden)?.to_dtype(DType::F32)?,
                    decay_input: through(forget)?
                        .reshape((batch, sequence, key_heads, key_dim))?
                        .to_dtype(DType::F32)?,
                })
            }
            Self::OlmoHybrid {
                query,
                key,
                value,
                strength,
                decay,
                gate,
            } => Ok(Prepared {
                mixed: Tensor::cat(
                    &[
                        &query.forward(hidden)?,
                        &key.forward(hidden)?,
                        &value.forward(hidden)?,
                    ],
                    2,
                )?,
                gate: gate
                    .forward(hidden)?
                    .reshape((batch, sequence, value_heads, value_dim))?,
                strength: strength.forward(hidden)?.to_dtype(DType::F32)?,
                decay_input: decay
                    .forward(hidden)?
                    .reshape((batch, sequence, value_heads, 1))?
                    .to_dtype(DType::F32)?,
            }),
            Self::Qwen35 {
                query_key_value,
                strength,
                decay,
                gate,
            } => Ok(Prepared {
                mixed: query_key_value.forward(hidden)?,
                gate: gate
                    .forward(hidden)?
                    .reshape((batch, sequence, value_heads, value_dim))?,
                strength: strength.forward(hidden)?.to_dtype(DType::F32)?,
                decay_input: decay
                    .forward(hidden)?
                    .reshape((batch, sequence, value_heads, 1))?
                    .to_dtype(DType::F32)?,
            }),
        }
    }
}
