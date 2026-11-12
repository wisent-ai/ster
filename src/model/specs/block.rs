//! A block's feed-forward kind, norm, rotary scaling, query and key norm and
//! activation.

use candle_core::Tensor;

/// Whether the feed-forward gates its up projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardKind {
    /// `down(act(gate(x)) * up(x))`.
    Gated,
    /// `down(act(up(x)))`.
    Plain,
}

/// How a family normalises, read from its config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Root mean square, no mean subtraction, no bias.
    Rms,
    /// Mean subtracted, then divided by the standard deviation; `bias` says
    /// whether a bias tensor follows the scale.
    Layer { bias: bool },
    /// LayerNorm with no scale and no bias at all (OLMo 1).
    Bare,
    /// Root mean square with no scale at all (HRM-Text's norms).
    BareRms,
}

/// A rotary scaling read from the config's `rope_scaling`, beyond Llama 3's.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    None,
    /// Every angle divided by the factor.
    Linear(f32),
    /// Phi-3's LongRoPE: each frequency divided by its own factor, from
    /// `short` while the sequence is within `original` positions and from
    /// `long` once it goes beyond; the short table is multiplied by
    /// `short_attention`, the long one by `long_attention` (equal for Phi-3,
    /// PhiMoE's `short_mscale` and `long_mscale` otherwise).
    LongRope {
        short: Vec<f32>,
        long: Vec<f32>,
        original: usize,
        short_attention: f32,
        long_attention: f32,
    },
    /// YaRN: frequencies whose wavelength fits `original` positions more
    /// than `beta_fast` times keep their value, those fitting fewer than
    /// `beta_slow` times are divided by `factor`, a linear ramp blends the
    /// band between, and both tables are multiplied by `attention`. The band's
    /// bounds are rounded out to whole dimension pairs unless `truncate` is
    /// false (Transformers' `truncate`, which GPT-OSS and PLaMo 3 turn off).
    Yarn {
        factor: f32,
        original: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention: f32,
        truncate: bool,
    },
    /// Gemma 4's proportional rotation: the first `rotated` frequency pairs
    /// keep their value and the rest are zero, so those components pass
    /// through unrotated; every frequency is divided by `factor`.
    Proportional {
        rotated: usize,
        factor: f32,
    },
}

/// Where a query and key norm sits, if the family has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKeyNorm {
    None,
    /// One norm over each head's `head_dim`, shared by every head (Qwen3,
    /// Gemma 3).
    PerHead,
    /// One norm over the whole projection (OLMo 2).
    Full,
    /// A separate scale per head, stored as one `[heads, head_dim]` tensor
    /// (Cohere's `use_qk_norm`).
    HeadWeights,
    /// A separate norm per head, stored as one module per head (StableLM's
    /// `qk_layernorm`).
    HeadModules,
    /// Llama 4's `use_qk_norm`: a weightless RMS norm over each head's
    /// query and key, after the rotation, on the rotating layers only.
    Unscaled,
    /// MuseGlimmer's QK-norm: a weightless RMS norm over each head's query
    /// and key on every layer, rotating or not.
    Weightless,
}

/// The non-linearity in the feed-forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Silu,
    /// `gelu_pytorch_tanh` and `gelu_new`, which Candle's `gelu` computes.
    GeluTanh,
    /// The exact, erf-based GELU.
    Gelu,
    /// Squared ReLU (Nemotron's `relu2`).
    Relu2,
    /// Plain ReLU (OPT).
    Relu,
    /// xIELU (Apertus), whose parameters are each feed-forward's own
    /// (`mlp.act_fn`); the feed-forward that holds them applies it.
    Xielu,
    /// Kimi-K3's SiTU (`situ`, `activation_situ_beta` and
    /// `activation_situ_linear_beta`): the gate's `beta · tanh(gate / beta) ·
    /// sigmoid(gate)` times the up projection's `linear_beta · tanh(up /
    /// linear_beta)` (the up projection itself without a `linear_beta`).
    Situ {
        beta: f64,
        linear_beta: Option<f64>,
    },
}

impl Activation {
    pub(crate) fn apply(self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Silu => candle_nn::ops::silu(input),
            Self::GeluTanh => input.gelu(),
            Self::Gelu => input.gelu_erf(),
            Self::Relu2 => input.relu()?.sqr(),
            Self::Relu => input.relu(),
            // The reading accepts xIELU only on Apertus, whose dense
            // feed-forward applies it with the parameters it holds.
            Self::Xielu => Err(crate::failures::refuse_candle(
                &crate::failures::model::decoder::XIELU_UNHELD,
                &[("shape", &format!("{:?}", input.dims()))],
            )),
            // sigmoid, composed so it has a backward pass.
            Self::Situ { beta, .. } => {
                ((input / beta)?.tanh()? * beta)? * (input.neg()?.exp()? + 1.0)?.recip()?
            }
        }
    }

    /// A gated feed-forward's product: the activated gate times the up
    /// projection, which only SiTU transforms first.
    pub(crate) fn gated(self, gate: &Tensor, up: &Tensor) -> candle_core::Result<Tensor> {
        let up = match self {
            Self::Situ {
                linear_beta: Some(linear_beta),
                ..
            } => ((up / linear_beta)?.tanh()? * linear_beta)?,
            _ => up.clone(),
        };
        self.apply(gate)? * up
    }
}
