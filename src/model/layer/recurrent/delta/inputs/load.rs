//! Loading each delta-rule form's input projections, convolution taps and
//! `dt_bias` from its own tensor names.

use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, linear_no_bias};

use super::Inputs;
use crate::model::{DeltaRuleForm, DeltaRuleSpec};

impl Inputs {
    /// The form's projections, its convolution taps `[channels, kernel]`,
    /// and its `dt_bias` shaped to add to the decay input, in F32.
    pub(in crate::model::layer::recurrent::delta) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: DeltaRuleSpec,
    ) -> candle_core::Result<(Self, Tensor, Tensor)> {
        let DeltaRuleSpec {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel,
            form,
            full_rank_gate,
            ..
        } = spec;
        let keys = key_heads * key_dim;
        let values = value_heads * value_dim;
        let taps = |name: &str, width: usize| -> candle_core::Result<Tensor> {
            builder
                .pp(name)
                .get((width, 1, kernel), "weight")?
                .reshape((width, kernel))
        };
        Ok(match form {
            DeltaRuleForm::Qwen3Next => (
                Self::Qwen3Next {
                    query_key_value_gate: linear_no_bias(
                        hidden,
                        2 * keys + 2 * values,
                        builder.pp("in_proj_qkvz"),
                    )?,
                    strength_decay: linear_no_bias(
                        hidden,
                        2 * value_heads,
                        builder.pp("in_proj_ba"),
                    )?,
                },
                taps("conv1d", 2 * keys + values)?,
                builder
                    .get(value_heads, "dt_bias")?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, value_heads, 1))?,
            ),
            DeltaRuleForm::Kimi | DeltaRuleForm::Ling => (
                Self::Kimi {
                    query: linear_no_bias(hidden, keys, builder.pp("q_proj"))?,
                    key: linear_no_bias(hidden, keys, builder.pp("k_proj"))?,
                    value: linear_no_bias(hidden, values, builder.pp("v_proj"))?,
                    strength: linear_no_bias(hidden, value_heads, builder.pp("b_proj"))?,
                    forget: if form == DeltaRuleForm::Ling {
                        (None, linear_no_bias(hidden, keys, builder.pp("f_proj"))?)
                    } else {
                        (
                            Some(linear_no_bias(hidden, key_dim, builder.pp("f_a_proj"))?),
                            linear_no_bias(key_dim, keys, builder.pp("f_b_proj"))?,
                        )
                    },
                    gate: if form == DeltaRuleForm::Ling || full_rank_gate {
                        (None, linear_no_bias(hidden, values, builder.pp("g_proj"))?)
                    } else {
                        (
                            Some(linear_no_bias(hidden, value_dim, builder.pp("g_a_proj"))?),
                            linear_no_bias(value_dim, values, builder.pp("g_b_proj"))?,
                        )
                    },
                },
                Tensor::cat(
                    &[
                        &taps("q_conv1d", keys)?,
                        &taps("k_conv1d", keys)?,
                        &taps("v_conv1d", values)?,
                    ],
                    0,
                )?,
                builder
                    .get(keys, "dt_bias")?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, key_heads, key_dim))?,
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
                    Tensor::cat(
                        &[
                            &taps("q_conv1d", keys)?,
                            &taps("k_conv1d", keys)?,
                            &taps("v_conv1d", values)?,
                        ],
                        0,
                    )?
                },
                builder
                    .get(value_heads, "dt_bias")?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, value_heads, 1))?,
            ),
            DeltaRuleForm::Qwen35 => (
                Self::Qwen35 {
                    query_key_value: linear_no_bias(
                        hidden,
                        2 * keys + values,
                        builder.pp("in_proj_qkv"),
                    )?,
                    strength: linear_no_bias(hidden, value_heads, builder.pp("in_proj_b"))?,
                    decay: linear_no_bias(hidden, value_heads, builder.pp("in_proj_a"))?,
                    gate: linear_no_bias(hidden, values, builder.pp("in_proj_z"))?,
                },
                taps("conv1d", 2 * keys + values)?,
                builder
                    .get(value_heads, "dt_bias")?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, value_heads, 1))?,
            ),
        })
    }
}
