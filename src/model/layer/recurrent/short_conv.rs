//! LFM2's gated short convolution, the mixer that replaces attention on its
//! convolution layers; it runs the causal depthwise convolution the
//! state-space mixers share.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::state_space::causal_convolution;
use crate::model::layer::projection;
use crate::model::{Cache, ShortConvolution};

/// LFM2's gated short convolution: `in_proj` yields `B`, `C` and `x`, the
/// causal convolution runs over `B·x`, and `C` times its output goes through
/// `out_proj`. Tensor names follow Transformers' `Lfm2ShortConv`:
/// `in_proj`, `conv` and `out_proj`.
#[derive(Debug, Clone)]
pub(in crate::model::layer) struct ShortConv {
    input: Linear,
    /// `[hidden, kernel]`.
    convolution: Tensor,
    convolution_bias: Option<Tensor>,
    output: Linear,
    kernel: usize,
}

impl ShortConv {
    /// `builder` is the mixer's (`model.layers.{i}.conv`).
    pub(in crate::model::layer) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        spec: &ShortConvolution,
    ) -> candle_core::Result<Self> {
        let convolution = builder.pp("conv");
        Ok(Self {
            input: projection(hidden, 3 * hidden, spec.bias, false, builder.pp("in_proj"))?,
            convolution: convolution
                .get((hidden, 1, spec.kernel), "weight")?
                .reshape((hidden, spec.kernel))?,
            convolution_bias: if spec.bias {
                Some(convolution.get(hidden, "bias")?)
            } else {
                None
            },
            output: projection(hidden, hidden, spec.bias, false, builder.pp("out_proj"))?,
            kernel: spec.kernel,
        })
    }

    /// Mixes `hidden` `[batch, sequence, width]`, continuing from the
    /// layer's saved convolution inputs when the cache keeps them.
    pub(in crate::model::layer) fn forward(
        &self,
        hidden: &Tensor,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let (_, _, width) = hidden.dims3()?;
        let dtype = hidden.dtype();
        let projected = self.input.forward(hidden)?;
        let gate_in = projected.narrow(2, 0, width)?;
        let gate_out = projected.narrow(2, width, width)?;
        let stream = projected.narrow(2, 2 * width, width)?;
        let saved = if cache.use_kv_cache {
            cache.states[layer].clone()
        } else {
            None
        };
        let (convolved, next_history) = causal_convolution(
            &(gate_in * stream)?.transpose(1, 2)?,
            saved.map(|(history, _)| history),
            &self.convolution,
            self.convolution_bias.as_ref(),
            self.kernel,
        )?;
        if cache.use_kv_cache {
            // The slot holds a pair; a convolution has nothing to scan, so
            // the second is the same history handle.
            cache.states[layer] = Some((next_history.clone(), next_history));
        }
        let gated = (gate_out.to_dtype(DType::F32)? * convolved.transpose(1, 2)?)?;
        self.output.forward(&gated.to_dtype(dtype)?)
    }
}
