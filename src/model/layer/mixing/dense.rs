//! A dense feed-forward: its gate, up and down projections, the activation
//! between them and the clamps and sparsity some families add.

use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::super::{
    Activation, Architecture, DenseClamp, FeedForwardKind, Route, attention::project,
};
use super::{experts::expert::swiglu_oai, projection};

#[derive(Debug, Clone)]
pub(super) struct FeedForward {
    /// `None` on a plain feed-forward (Starcoder2, Phi-2, Nemotron).
    gate: Option<Linear>,
    up: Linear,
    down: Linear,
    gate_adapter: Option<Adapter>,
    up_adapter: Option<Adapter>,
    down_adapter: Option<Adapter>,
    activation: Activation,
    /// Falcon-H1's `mlp_multipliers`: on the gate before its activation,
    /// and on the output.
    scales: Option<(f64, f64)>,
    /// xIELU's parameters, when the activation is xIELU (Apertus).
    xielu: Option<Xielu>,
    /// Gemma 3n's activation sparsity: the gate keeps only what lies above
    /// its mean plus this many standard deviations.
    sparsity: Option<f64>,
    /// The clamped SwiGLU of the dense feed-forward: GLM-5-Next's gate
    /// clamped above and up projection both ways before they meet, or
    /// MiniMax-M3's SwiGLU-OAI.
    clamp: Option<DenseClamp>,
}

/// xIELU's parameters from the feed-forward's `act_fn`, as Transformers'
/// `XIELUActivation` uses them: `alpha_p` is the softplus of the stored
/// one, `alpha_n` is `beta` plus the softplus of the stored one.
#[derive(Debug, Clone, Copy)]
struct Xielu {
    alpha_p: f64,
    alpha_n: f64,
    beta: f64,
    eps: f64,
}

impl Xielu {
    fn load(builder: VarBuilder<'_>) -> candle_core::Result<Self> {
        let scalar = |name: &str| -> candle_core::Result<f64> {
            let stored = builder.get_unchecked(name)?;
            let values = stored
                .flatten_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_vec1::<f64>()?;
            match values.as_slice() {
                [value] => Ok(*value),
                _ => Err(crate::model::misshapen(&format!("xIELU's {name}"), (1,), stored.shape())),
            }
        };
        let softplus = |value: f64| value.exp().ln_1p();
        let beta = scalar("beta")?;
        Ok(Self {
            alpha_p: softplus(scalar("alpha_p")?),
            alpha_n: beta + softplus(scalar("alpha_n")?),
            beta,
            eps: scalar("eps")?,
        })
    }

    /// `alpha_p · x² + beta · x` above zero, and
    /// `(expm1(min(x, eps)) − x) · alpha_n + beta · x` elsewhere.
    fn apply(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let linear = (input * self.beta)?;
        let positive = ((input.sqr()? * self.alpha_p)? + &linear)?;
        let negative =
            ((((input.minimum(self.eps)?.exp()? - 1.0)? - input)? * self.alpha_n)? + &linear)?;
        input.gt(0.0)?.where_cond(&positive, &negative)
    }
}

impl FeedForward {
    /// `builder` is the layer's, so the family's names (`mlp.up_proj`,
    /// `mlp.c_fc`, `mlp.fc1`) are read from the architecture whole.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        // DeciLM states each layer's feed-forward width apart.
        let intermediate = match architecture
            .layer_plan(layer)
            .and_then(|plan| plan.intermediate)
        {
            Some(own) => own,
            None => config.intermediate_size,
        };
        let hidden = config.hidden_size;
        let bias = architecture.feed_forward_bias;
        let names = architecture.names;
        let conv1d = architecture.conv1d;
        // Phi-3 and GLM store the gate and up projections as one matrix, gate
        // rows first; each is a row slice of that mapped weight.
        let (gate, up) = if architecture.fused_feed_forward {
            let fused = builder.pp(names.fused_gate_up);
            let weight = fused.get((2 * intermediate, hidden), "weight")?;
            let bias = if bias {
                Some(fused.get(2 * intermediate, "bias")?)
            } else {
                None
            };
            let slice = |start: usize| -> candle_core::Result<Linear> {
                Ok(Linear::new(
                    weight.narrow(0, start, intermediate)?,
                    bias.as_ref()
                        .map(|bias| bias.narrow(0, start, intermediate))
                        .transpose()?,
                ))
            };
            (Some(slice(0)?), slice(intermediate)?)
        } else {
            let gate = match (architecture.feed_forward, names.gate) {
                (FeedForwardKind::Gated, Some(name)) => Some(projection(
                    hidden,
                    intermediate,
                    bias,
                    conv1d,
                    builder.pp(name),
                )?),
                _ => None,
            };
            (
                gate,
                projection(hidden, intermediate, bias, conv1d, builder.pp(names.up))?,
            )
        };
        Ok(Self {
            gate,
            up,
            down: projection(
                intermediate,
                hidden,
                bias || architecture.down_bias,
                conv1d,
                builder.pp(names.down),
            )?,
            gate_adapter: adapters.get(layer, Target::Gate).cloned(),
            up_adapter: adapters.get(layer, Target::Up).cloned(),
            down_adapter: adapters.get(layer, Target::Down).cloned(),
            activation: architecture.activation,
            scales: architecture.feed_forward_scales,
            xielu: if architecture.activation == Activation::Xielu {
                let block = match names.down.rsplit_once('.') {
                    Some((parent, _)) if !parent.is_empty() => builder.pp(parent),
                    _ => builder.clone(),
                };
                Some(Xielu::load(block.pp("act_fn"))?)
            } else {
                None
            },
            sparsity: architecture
                .activation_sparsity
                .filter(|(_, layers)| layer < u128::BITS as usize && layers & (1u128 << layer) != 0)
                .map(|(multiplier, _)| multiplier),
            clamp: architecture.dense_swiglu_limit,
        })
    }

    /// No `Pass` here — every activation and the elementwise product
    /// backpropagate, so the feed-forward block is already differentiable as
    /// written — but a `Route`, because its projections are adapter sites
    /// like any other.
    pub(super) fn forward(&self, hidden: &Tensor, route: Route) -> candle_core::Result<Tensor> {
        let up = project(&self.up, self.up_adapter.as_ref(), hidden, route)?;
        let inner = match &self.gate {
            Some(gate) => {
                let gate = project(gate, self.gate_adapter.as_ref(), hidden, route)?;
                let gate = match self.scales {
                    Some((gate_scale, _)) => (gate * gate_scale)?,
                    None => gate,
                };
                let gate = self.sparse(gate)?;
                match (self.clamp, &self.xielu) {
                    (Some(DenseClamp::Oai { limit, alpha }), _) => {
                        swiglu_oai(gate, up, limit, alpha)?
                    }
                    (Some(DenseClamp::Inner(limit)), _) => self
                        .activation
                        .gated(&gate.minimum(limit)?, &up.clamp(-limit, limit)?)?,
                    (None, Some(xielu)) => (xielu.apply(&gate)? * up)?,
                    (None, None) => self.activation.gated(&gate, &up)?,
                }
            }
            None => self.activate(&up)?,
        };
        let output = project(&self.down, self.down_adapter.as_ref(), &inner, route)?;
        match self.scales {
            Some((_, output_scale)) => output * output_scale,
            None => Ok(output),
        }
    }

    fn activate(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        match &self.xielu {
            Some(xielu) => xielu.apply(input),
            None => self.activation.apply(input),
        }
    }

    /// Gemma 3n's Gaussian top-k (`Gemma3nTextMLP._gaussian_topk`):
    /// `relu(gate − mean − std · multiplier)` per position, the standard
    /// deviation taken over the whole width; the gate unchanged elsewhere.
    fn sparse(&self, gate: Tensor) -> candle_core::Result<Tensor> {
        let Some(multiplier) = self.sparsity else {
            return Ok(gate);
        };
        let mean = gate.mean_keepdim(candle_core::D::Minus1)?;
        let deviation = gate
            .broadcast_sub(&mean)?
            .sqr()?
            .mean_keepdim(candle_core::D::Minus1)?
            .sqrt()?;
        gate.broadcast_sub(&(mean + (deviation * multiplier)?)?)?
            .relu()
    }
}

