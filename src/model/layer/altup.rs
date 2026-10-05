//! Gemma 3n's alternating updates (AltUp) and learned augmented residual
//! (Laurel), as Transformers' `modeling_gemma3n.py` computes them.
//!
//! AltUp carries `streams` copies of the hidden state through the stack;
//! the blocks run on the first (the one Ster steers and reads), and each
//! layer predicts every stream from all of them before its block and
//! corrects every stream by what the block did to the first after it.

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::norm::{Norm, NormSpec};
use crate::model::Pass;

/// The floor under a stream's mean square before its magnitude is matched to
/// the first stream's (`Gemma3nTextModel`, `epsilon_tensor = 1e-5`).
const MAGNITUDE_FLOOR: f64 = 1e-5;

/// One layer's AltUp: `altup.prediction_coefs`, `correction_coefs`,
/// `modality_router` behind `router_norm`, and `correct_output_scale`.
#[derive(Debug, Clone)]
pub(in crate::model) struct AltUp {
    prediction: Linear,
    correction: Linear,
    router: Linear,
    router_norm: Norm,
    output_scale: Tensor,
    streams: usize,
    hidden: usize,
}

impl AltUp {
    pub(in crate::model) fn load(
        builder: VarBuilder<'_>,
        hidden: usize,
        streams: usize,
        norms: NormSpec,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            prediction: linear_no_bias(streams, streams * streams, builder.pp("prediction_coefs"))?,
            correction: linear_no_bias(streams, streams, builder.pp("correction_coefs"))?,
            router: linear_no_bias(hidden, streams, builder.pp("modality_router"))?,
            router_norm: norms.load(hidden, builder.pp("router_norm"))?,
            output_scale: builder.get(hidden, "correct_output_scale")?,
            streams,
            hidden,
        })
    }

    /// `tanh(modality_router(router_norm(x) / hidden_size))`, one weight per
    /// stream, in F32.
    fn modalities(&self, active: &Tensor, pass: Pass) -> candle_core::Result<Tensor> {
        let routed = self.router.forward(&(self.router_norm.forward(active, pass)? / self.hidden as f64)?)?;
        routed.to_dtype(DType::F32)?.tanh()
    }

    /// Every stream's prediction: stream `i` plus `Σ_j C[i, j] · stream j`,
    /// `C` the `[streams, streams]` coefficients the first stream routes to.
    pub(in crate::model) fn predict(&self, streams: &[Tensor], pass: Pass) -> candle_core::Result<Vec<Tensor>> {
        let (batch, sequence, _) = streams[0].dims3()?;
        let coefficients = self
            .prediction
            .forward(&self.modalities(&streams[0], pass)?.to_dtype(streams[0].dtype())?)?
            .reshape((batch, sequence, self.streams, self.streams))?;
        (0..self.streams)
            .map(|row| {
                let mut prediction = streams[row].clone();
                for (column, stream) in streams.iter().enumerate() {
                    let coefficient = coefficients.narrow(2, row, 1)?.narrow(3, column, 1)?.squeeze(3)?;
                    prediction = (prediction + stream.broadcast_mul(&coefficient)?)?;
                }
                Ok(prediction)
            })
            .collect()
    }

    /// Every stream corrected by the innovation `activated − prediction₀`,
    /// times `correction_coefs(modalities(activated)) + 1` for that stream.
    pub(in crate::model) fn correct(
        &self,
        predictions: &[Tensor],
        activated: &Tensor,
        pass: Pass,
    ) -> candle_core::Result<Vec<Tensor>> {
        let innovation = (activated - &predictions[0])?;
        let coefficients =
            (self.correction.forward(&self.modalities(activated, pass)?.to_dtype(activated.dtype())?)? + 1.0)?;
        predictions
            .iter()
            .enumerate()
            .map(|(stream, prediction)| prediction + innovation.broadcast_mul(&coefficients.narrow(2, stream, 1)?)?)
            .collect()
    }

    /// The first corrected stream times `correct_output_scale`, which the
    /// per-layer input gate reads.
    pub(in crate::model) fn scaled(&self, corrected: &Tensor) -> candle_core::Result<Tensor> {
        corrected.broadcast_mul(&self.output_scale.to_dtype(corrected.dtype())?)
    }
}

/// Laurel: `normed + post_laurel_norm(linear_right(linear_left(normed)))`.
#[derive(Debug, Clone)]
pub(in crate::model) struct Laurel {
    left: Linear,
    right: Linear,
    norm: Norm,
}

impl Laurel {
    pub(in crate::model) fn load(builder: VarBuilder<'_>, hidden: usize, rank: usize, norms: NormSpec) -> candle_core::Result<Self> {
        Ok(Self {
            left: linear_no_bias(hidden, rank, builder.pp("linear_left"))?,
            right: linear_no_bias(rank, hidden, builder.pp("linear_right"))?,
            norm: norms.load(hidden, builder.pp("post_laurel_norm"))?,
        })
    }

    pub(in crate::model) fn forward(&self, normed: &Tensor, pass: Pass) -> candle_core::Result<Tensor> {
        normed + self.norm.forward(&self.right.forward(&self.left.forward(normed)?)?, pass)?
    }
}

/// The model-level projections into the extra streams after the embedding
/// (`altup_projections`) and out of them before the final norm
/// (`altup_unembed_projections`), each stream's magnitude matched to the
/// first's.
#[derive(Debug, Clone)]
pub(in crate::model) struct StreamProjections {
    into: Vec<Linear>,
    out: Vec<Linear>,
}

impl StreamProjections {
    pub(in crate::model) fn load(builder: &VarBuilder<'_>, hidden: usize, streams: usize) -> candle_core::Result<Self> {
        let all = |name: &str| -> candle_core::Result<Vec<Linear>> {
            (0..streams - 1).map(|index| linear_no_bias(hidden, hidden, builder.pp(format!("{name}.{index}")))).collect()
        };
        Ok(Self { into: all("altup_projections")?, out: all("altup_unembed_projections")? })
    }

    /// The extra streams the first one starts beside.
    pub(in crate::model) fn spread(&self, first: &Tensor) -> candle_core::Result<Vec<Tensor>> {
        self.into.iter().map(|projection| matched(&projection.forward(first)?, first)).collect()
    }

    /// The streams joined back into one: each extra stream projected out and
    /// matched to the first's magnitude, then all averaged.
    pub(in crate::model) fn join(&self, first: &Tensor, rest: &[Tensor]) -> candle_core::Result<Tensor> {
        let mut sum = first.clone();
        for (projection, stream) in self.out.iter().zip(rest) {
            sum = (sum + matched(&projection.forward(stream)?, first)?)?;
        }
        sum / (rest.len() + 1) as f64
    }
}

/// `stream` rescaled to the root-mean-square of `target`, per position.
fn matched(stream: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    let target_magnitude = target.sqr()?.mean_keepdim(D::Minus1)?.sqrt()?;
    let magnitude = stream.sqr()?.mean_keepdim(D::Minus1)?.maximum(MAGNITUDE_FLOOR)?.sqrt()?;
    stream.broadcast_mul(&(target_magnitude / magnitude)?)
}
