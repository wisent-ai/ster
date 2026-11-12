//! One expert of a mixture: its projections and the clamped SwiGLU some
//! families run it with.

use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::Activation;

#[derive(Debug, Clone)]
pub(super) struct Expert {
    /// `None` in Nemotron-H's experts, which are `down(act(up))`.
    pub(super) gate: Option<Linear>,
    pub(super) up: Linear,
    /// `None` in K2-Horizon's value experts, which are `act(up)`.
    pub(super) down: Option<Linear>,
}

/// GPT-OSS's gate sharpness: the gate is `g · sigmoid(1.702 · g)`, the
/// sigmoid approximation of GELU it was trained with.
pub(super) const GPT_OSS_GATE_SHARPNESS: f64 = 1.702;

/// One expert's clamped SwiGLU, resolved for its layer.
#[derive(Debug, Clone, Copy)]
pub(super) enum Clamp {
    /// GPT-OSS's `(clamp(up, -limit, limit) + 1) · g · sigmoid(alpha · g)`
    /// where `g = min(gate, limit)`: [`GPT_OSS_GATE_SHARPNESS`] for GPT-OSS,
    /// `swiglu_alpha` for MiniMax-M3.
    GptOss { limit: f64, alpha: f64 },
    /// Step 3.5's `min(act(gate), limit) · clamp(up, -limit, limit)`.
    Step(f64),
    /// DeepSeek-V4's `act(min(gate, limit)) · clamp(up, -limit, limit)`.
    Inner(f64),
}

impl Expert {
    /// `down(act(gate) · up)`, or `down(act(up))` without a gate, with the
    /// gate and up clamped as `clamp` says.
    pub(super) fn forward(
        &self,
        input: &Tensor,
        activation: Activation,
        clamp: Option<Clamp>,
    ) -> candle_core::Result<Tensor> {
        let up = self.up.forward(input)?;
        let gated = match (&self.gate, clamp) {
            (Some(gate), Some(Clamp::GptOss { limit, alpha })) => {
                swiglu_oai(gate.forward(input)?, up, limit, alpha)?
            }
            (Some(gate), Some(Clamp::Step(limit))) => {
                let gate = activation.apply(&gate.forward(input)?)?.minimum(limit)?;
                (gate * up.clamp(-limit, limit)?)?
            }
            (Some(gate), Some(Clamp::Inner(limit))) => {
                let gate = activation.apply(&gate.forward(input)?.minimum(limit)?)?;
                (gate * up.clamp(-limit, limit)?)?
            }
            (Some(gate), None) => activation.gated(&gate.forward(input)?, &up)?,
            (None, _) => activation.apply(&up)?,
        };
        match &self.down {
            Some(down) => down.forward(&gated),
            None => Ok(gated),
        }
    }

    /// An expert of three unbiased projections named `[gate, up, down]`, or
    /// of `[up, down]` alone when `gate` is `None`.
    pub(super) fn load(
        hidden: usize,
        intermediate: usize,
        gate: Option<&str>,
        [up, down]: [&str; 2],
        builder: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            gate: gate
                .map(|gate| linear_no_bias(hidden, intermediate, builder.pp(gate)))
                .transpose()?,
            up: linear_no_bias(hidden, intermediate, builder.pp(up))?,
            down: Some(linear_no_bias(intermediate, hidden, builder.pp(down))?),
        })
    }
}

/// GPT-OSS's clamped SwiGLU, which MiniMax-M3 calls SwiGLU-OAI:
/// `(clamp(up, ±limit) + 1) · g · sigmoid(alpha · g)` with
/// `g = min(gate, limit)`, the sigmoid written out so it has a backward pass.
pub(in crate::model::layer) fn swiglu_oai(
    gate: Tensor,
    up: Tensor,
    limit: f64,
    alpha: f64,
) -> candle_core::Result<Tensor> {
    let gate = gate.minimum(limit)?;
    let up = (up.clamp(-limit, limit)? + 1.0)?;
    let sigmoid = ((gate.clone() * -alpha)?.exp()? + 1.0)?.recip()?;
    (gate * sigmoid)? * up
}

