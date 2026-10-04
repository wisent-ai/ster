//! Mamba-2's structured state-space mixer (the SSD layer), in place of
//! attention.
//!
//! One projection yields the gate `z`, the stream `x` with the input and
//! output matrices `B` and `C`, and a step per head. `x`, `B` and `C` pass
//! through a causal depthwise convolution and SiLU; each head then scans its
//! `[head_dim, state]` state with one scalar decay `exp(Δ·A)`, reading `B` and
//! `C` from its group. The read-out plus `D·x` is gated by SiLU of `z`, RMS
//! normalised per group with a stored scale, and projected back. Tensor names
//! follow Transformers' `Mamba2Mixer`: `in_proj`, `conv1d`, `dt_bias`,
//! `A_log`, `D`, `norm` and `out_proj`.
//!
//! The decode state is the convolution's last `conv_kernel - 1` inputs and
//! the scan state, kept in the cache's key-value slot for the layer.

use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::model::{Cache, StateSpaceSpec, StructuredSpec};

use super::{projection, state_space::softplus};

#[derive(Debug, Clone)]
pub(super) struct Structured {
    input: Linear,
    /// The depthwise convolution over `x`, `B` and `C`, `[channels, kernel]`.
    convolution: Tensor,
    convolution_bias: Option<Tensor>,
    /// `dt_bias`, `[heads]`, in F32.
    step_bias: Tensor,
    /// `-exp(A_log)`, `[heads]`, in F32.
    decay: Tensor,
    /// `D`, `[heads]`, in F32.
    skip: Tensor,
    /// The gated norm's scale, `[inner]`.
    norm: Tensor,
    eps: f64,
    output: Linear,
    spec: StateSpaceSpec,
    heads: StructuredSpec,
}

impl Structured {
    /// `builder` is the mixer's (`backbone.layers.{i}.mixer`, or Bamba's
    /// `model.layers.{i}.mamba`).
    pub(super) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        eps: f64,
        spec: &StateSpaceSpec,
        heads: StructuredSpec,
    ) -> candle_core::Result<Self> {
        let StateSpaceSpec {
            inner,
            state,
            kernel,
            projection_bias,
            convolution_bias,
            ..
        } = *spec;
        let channels = inner + 2 * heads.groups * state;
        let projected = inner + channels + heads.heads;
        let convolution = builder.pp("conv1d");
        let f32_vector = |name: &str| -> candle_core::Result<Tensor> {
            builder.get(heads.heads, name)?.to_dtype(DType::F32)
        };
        Ok(Self {
            input: projection(hidden, projected, projection_bias, false, builder.pp("in_proj"))?,
            convolution: convolution
                .get((channels, 1, kernel), "weight")?
                .reshape((channels, kernel))?,
            convolution_bias: if convolution_bias {
                Some(convolution.get(channels, "bias")?)
            } else {
                None
            },
            step_bias: f32_vector("dt_bias")?,
            decay: f32_vector("A_log")?.exp()?.neg()?,
            skip: f32_vector("D")?,
            norm: builder.pp("norm").get(inner, "weight")?,
            eps,
            output: projection(inner, hidden, projection_bias, false, builder.pp("out_proj"))?,
            spec: *spec,
            heads,
        })
    }

    /// Mixes `hidden` `[batch, sequence, width]`, continuing from the
    /// layer's saved state when the cache keeps one. Every op has a backward
    /// pass, so the same code serves inference and training.
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
            ..
        } = self.spec;
        let StructuredSpec {
            heads,
            head_dim,
            groups,
            step_limit,
        } = self.heads;
        let channels = inner + 2 * groups * state;
        let dtype = hidden.dtype();
        let device = hidden.device();
        let projected = self.input.forward(hidden)?;
        let gate = projected.narrow(2, 0, inner)?;
        let mixed = projected.narrow(2, inner, channels)?;
        let step = projected.narrow(2, inner + channels, heads)?.to_dtype(DType::F32)?;

        // The causal depthwise convolution over `x`, `B` and `C`, continued
        // from the inputs the previous call ended on.
        let mixed = mixed.transpose(1, 2)?.to_dtype(DType::F32)?;
        let saved = if cache.use_kv_cache { cache.kvs[layer].clone() } else { None };
        let (history, scan) = match saved {
            Some((history, scan)) => (history, scan),
            None => (
                Tensor::zeros((batch, channels, kernel - 1), DType::F32, device)?,
                Tensor::zeros((batch, heads, head_dim, state), DType::F32, device)?,
            ),
        };
        let padded = Tensor::cat(&[&history, &mixed], 2)?;
        let taps = self.convolution.to_dtype(DType::F32)?;
        let mut convolved = Tensor::zeros((batch, channels, sequence), DType::F32, device)?;
        for tap in 0..kernel {
            let weight = taps.narrow(1, tap, 1)?.reshape((1, channels, 1))?;
            convolved = (convolved + padded.narrow(2, tap, sequence)?.broadcast_mul(&weight)?)?;
        }
        if let Some(bias) = &self.convolution_bias {
            convolved =
                convolved.broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, channels, 1))?)?;
        }
        let next_history = padded.narrow(2, sequence, kernel - 1)?.contiguous()?;
        let mixed = candle_nn::ops::silu(&convolved)?.transpose(1, 2)?.contiguous()?;
        let stream = mixed.narrow(2, 0, inner)?;
        let input_matrix = mixed.narrow(2, inner, groups * state)?;
        let output_matrix = mixed.narrow(2, inner + groups * state, groups * state)?;

        // The step per head: softplus of the projection plus `dt_bias`,
        // clamped to `time_step_limit`.
        let step = softplus(&step.broadcast_add(&self.step_bias.unsqueeze(0)?.unsqueeze(0)?)?)?;
        let (low, high) = step_limit;
        let step = if low > 0.0 || high.is_finite() {
            step.clamp(low, high)?
        } else {
            step
        };

        // Each head reads its group's `B` and `C`.
        let per_group = heads / groups;
        let by_head = |matrix: Tensor| -> candle_core::Result<Tensor> {
            matrix
                .reshape((batch, groups, 1, state))?
                .broadcast_as((batch, groups, per_group, state))?
                .reshape((batch, heads, state))
        };
        let mut scan = scan;
        let mut outputs = Vec::with_capacity(sequence);
        for position in 0..sequence {
            let delta = step.i((.., position, ..))?;
            let token = stream.i((.., position, ..))?.reshape((batch, heads, head_dim))?;
            let written = by_head(input_matrix.i((.., position, ..))?)?;
            let read = by_head(output_matrix.i((.., position, ..))?)?;
            let decay = delta
                .broadcast_mul(&self.decay.unsqueeze(0)?)?
                .exp()?
                .reshape((batch, heads, 1, 1))?;
            let update = token
                .broadcast_mul(&delta.unsqueeze(2)?)?
                .unsqueeze(3)?
                .broadcast_mul(&written.unsqueeze(2)?)?;
            scan = (scan.broadcast_mul(&decay)? + update)?;
            let read_out = scan.broadcast_mul(&read.unsqueeze(2)?)?.sum(3)?;
            let skipped = token.broadcast_mul(&self.skip.reshape((1, heads, 1))?)?;
            outputs.push((read_out + skipped)?.reshape((batch, inner))?);
        }
        let scanned = Tensor::stack(&outputs, 1)?;
        if cache.use_kv_cache {
            cache.kvs[layer] = Some((next_history, scan));
        }

        // SiLU of the gate, then an RMS norm over each group's share of the
        // inner width, scaled by the stored weight.
        let gated = (scanned * candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?;
        let grouped = gated.reshape((batch, sequence, groups, inner / groups))?;
        let width = (inner / groups) as f64;
        let square = (grouped.sqr()?.sum_keepdim(D::Minus1)? / width)?;
        let normed = grouped
            .broadcast_div(&(square + self.eps)?.sqrt()?)?
            .reshape((batch, sequence, inner))?
            .to_dtype(dtype)?
            .broadcast_mul(&self.norm)?;
        self.output.forward(&normed)
    }
}
