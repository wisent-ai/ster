//! Mamba's selective state-space mixer, in place of attention.
//!
//! The block projects the normalised input to an inner width twice — a
//! stream and a gate — runs the stream through a short causal depthwise
//! convolution and SiLU, derives a step size and the input and output
//! matrices of a diagonal state-space model from it, scans it token by token,
//! and gates the result with SiLU of the gate before projecting back. Tensor
//! names follow Transformers' `MambaMixer`: `in_proj`, `conv1d`, `x_proj`,
//! `dt_proj`, `A_log`, `D` and `out_proj`.
//!
//! The decode state is the convolution's last `conv_kernel - 1` inputs and
//! the scan state, kept in the cache's key-value slot for the layer.

use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::model::{Cache, NormKind, ParameterNorm, Pass, StateSpaceSpec};

use super::{
    norm::{Norm, NormSpec},
    projection,
};

#[derive(Debug, Clone)]
pub(super) struct StateSpace {
    input: Linear,
    /// The depthwise convolution's taps, `[inner, kernel]`, and its bias.
    convolution: Tensor,
    convolution_bias: Option<Tensor>,
    parameters: Linear,
    step: Linear,
    /// `-exp(A_log)`, `[inner, state]`, in F32.
    decay: Tensor,
    /// `D`, `[inner]`, in F32.
    skip: Tensor,
    output: Linear,
    /// Jamba's `dt_layernorm`, `b_layernorm` and `c_layernorm`.
    parameter_norms: Option<[Norm; 3]>,
    spec: StateSpaceSpec,
}

impl StateSpace {
    /// `builder` is the mixer's (`backbone.layers.{i}.mixer`, or Jamba's
    /// `model.layers.{i}.mamba`).
    pub(super) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        eps: f64,
        spec: &StateSpaceSpec,
    ) -> candle_core::Result<Self> {
        let StateSpaceSpec {
            inner,
            state,
            kernel,
            step_rank,
            projection_bias,
            convolution_bias,
            ..
        } = *spec;
        let convolution = builder.pp("conv1d");
        Ok(Self {
            input: projection(hidden, 2 * inner, projection_bias, false, builder.pp("in_proj"))?,
            convolution: convolution.get((inner, 1, kernel), "weight")?.reshape((inner, kernel))?,
            convolution_bias: if convolution_bias {
                Some(convolution.get(inner, "bias")?)
            } else {
                None
            },
            parameters: projection(inner, step_rank + 2 * state, false, false, builder.pp("x_proj"))?,
            step: projection(step_rank, inner, true, false, builder.pp("dt_proj"))?,
            decay: builder.get((inner, state), "A_log")?.to_dtype(DType::F32)?.exp()?.neg()?,
            skip: builder.get(inner, "D")?.to_dtype(DType::F32)?,
            output: projection(inner, hidden, projection_bias, false, builder.pp("out_proj"))?,
            parameter_norms: match spec.parameter_norm {
                ParameterNorm::Weighted(norm_eps) => {
                    let norms = NormSpec {
                        kind: NormKind::Rms,
                        eps: if norm_eps > 0.0 { norm_eps } else { eps },
                        offset: false,
                    };
                    Some([
                        norms.load(step_rank, builder.pp("dt_layernorm"))?,
                        norms.load(state, builder.pp("b_layernorm"))?,
                        norms.load(state, builder.pp("c_layernorm"))?,
                    ])
                }
                _ => None,
            },
            spec: *spec,
        })
    }

    /// Mixes `hidden` `[batch, sequence, width]`, continuing from the
    /// layer's saved state when the cache keeps one.
    ///
    /// Every step is composed of ops with a backward pass, so the same code
    /// serves inference and training.
    pub(super) fn forward(
        &self,
        hidden: &Tensor,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let StateSpaceSpec {
            inner,
            state,
            kernel,
            step_rank,
            parameter_norm,
            ..
        } = self.spec;
        let dtype = hidden.dtype();
        let projected = self.input.forward(hidden)?;
        let stream = projected.narrow(2, 0, inner)?;
        let gate = projected.narrow(2, inner, inner)?;

        // The causal depthwise convolution, continued from the inputs the
        // previous call ended on.
        let stream = stream.transpose(1, 2)?.to_dtype(DType::F32)?;
        let saved = if cache.use_kv_cache { cache.kvs[layer].clone() } else { None };
        let (history, scan) = match saved {
            Some((history, scan)) => (history, scan),
            None => (
                Tensor::zeros((batch, inner, kernel - 1), DType::F32, hidden.device())?,
                Tensor::zeros((batch, inner, state), DType::F32, hidden.device())?,
            ),
        };
        let padded = Tensor::cat(&[&history, &stream], 2)?;
        let taps = self.convolution.to_dtype(DType::F32)?;
        let mut convolved = Tensor::zeros((batch, inner, sequence), DType::F32, hidden.device())?;
        for tap in 0..kernel {
            let weight = taps.narrow(1, tap, 1)?.reshape((1, inner, 1))?;
            convolved = (convolved + padded.narrow(2, tap, sequence)?.broadcast_mul(&weight)?)?;
        }
        if let Some(bias) = &self.convolution_bias {
            convolved = convolved.broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, inner, 1))?)?;
        }
        let next_history = padded.narrow(2, sequence, kernel - 1)?.contiguous()?;
        let stream = candle_nn::ops::silu(&convolved)?.transpose(1, 2)?.contiguous()?;

        // The step size and the input and output matrices, per token.
        let parameters = self.parameters.forward(&stream.to_dtype(dtype)?)?.to_dtype(DType::F32)?;
        let normalise = |part: Tensor, which: usize| -> candle_core::Result<Tensor> {
            match (parameter_norm, &self.parameter_norms) {
                (ParameterNorm::Bare(eps), _) => bare_rms(&part, eps),
                // The stored scale is at the weights' dtype; the norm widens
                // to F32 inside and the result returns to F32 for the scan.
                (ParameterNorm::Weighted(_), Some(norms)) => norms[which]
                    .forward(&part.to_dtype(dtype)?.contiguous()?, Pass::Differentiable)?
                    .to_dtype(DType::F32),
                _ => Ok(part),
            }
        };
        let step = normalise(parameters.narrow(2, 0, step_rank)?, 0)?;
        let input_matrix = normalise(parameters.narrow(2, step_rank, state)?, 1)?;
        let output_matrix = normalise(parameters.narrow(2, step_rank + state, state)?, 2)?;
        let step = softplus(&self.step.forward(&step.to_dtype(dtype)?)?.to_dtype(DType::F32)?)?;

        // The selective scan: state ← exp(Δ·A) ⊙ state + Δ·B·x, y = state·C + D·x.
        let mut scan = scan;
        let mut outputs = Vec::with_capacity(sequence);
        for position in 0..sequence {
            let delta = step.i((.., position, ..))?.unsqueeze(2)?;
            let token = stream.i((.., position, ..))?;
            let written = input_matrix.i((.., position, ..))?.unsqueeze(1)?;
            let read = output_matrix.i((.., position, ..))?.unsqueeze(1)?;
            let decay = delta.broadcast_mul(&self.decay.unsqueeze(0)?)?.exp()?;
            let update = delta
                .broadcast_mul(&written)?
                .broadcast_mul(&token.unsqueeze(2)?)?;
            scan = ((scan * decay)? + update)?;
            let read_out = scan.broadcast_mul(&read)?.sum(2)?;
            outputs.push((read_out + token.broadcast_mul(&self.skip.unsqueeze(0)?)?)?);
        }
        let scanned = Tensor::stack(&outputs, 1)?;
        if cache.use_kv_cache {
            cache.kvs[layer] = Some((next_history, scan));
        }
        let gated = (scanned * candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?;
        self.output.forward(&gated.to_dtype(dtype)?)
    }
}

/// `log(1 + exp(x))`, written as `max(x, 0) + log(1 + exp(-|x|))` so a large
/// step logit neither overflows nor loses its gradient.
pub(super) fn softplus(input: &Tensor) -> candle_core::Result<Tensor> {
    input.relu()? + (input.abs()?.neg()?.exp()? + 1.0)?.log()?
}

/// An RMS norm with no scale, as Falcon-Mamba applies to the step and the
/// input and output matrices.
fn bare_rms(input: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let width = input.dim(D::Minus1)? as f64;
    let square = (input.sqr()?.sum_keepdim(D::Minus1)? / width)?;
    input.broadcast_div(&(square + eps)?.sqrt()?)
}
