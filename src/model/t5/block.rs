//! T5's pre-normalized self-attention, decoder cross-attention and feed-forward.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, VarBuilder, kv_cache::KvCache, linear_no_bias};

use crate::{
    lora::{Adapter, Adapters, Target},
    model::{Activation, Mode, Route, attention::project, layer::norm::Norm},
};

use super::{attention::{Attention, KeyValue}, config::Config};

#[derive(Debug, Clone)]
pub(super) struct Block {
    attention: Attention,
    attention_norm: Norm,
    cross: Option<(Attention, Norm)>,
    feed_forward: FeedForward,
}

#[derive(Debug, Clone)]
struct FeedForward {
    norm: Norm,
    gate: Option<Linear>,
    up: Linear,
    down: Linear,
    gate_adapter: Option<Adapter>,
    up_adapter: Option<Adapter>,
    down_adapter: Option<Adapter>,
    activation: Activation,
}

impl Block {
    pub fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        layer: usize,
        decoder: bool,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let site = decoder.then_some((layer, adapters));
        let self_layer = builder.pp("layer.0");
        let attention = Attention::load(self_layer.pp("SelfAttention"), config, site)?;
        let attention_norm = config.norm().load(config.hidden, self_layer.pp("layer_norm"))?;
        let cross = if decoder {
            let cross_layer = builder.pp("layer.1");
            Some((
                Attention::load(cross_layer.pp("EncDecAttention"), config, None)?,
                config.norm().load(config.hidden, cross_layer.pp("layer_norm"))?,
            ))
        } else {
            None
        };
        let feed_forward = FeedForward::load(
            builder.pp(if decoder { "layer.2" } else { "layer.1" }),
            config,
            site,
        )?;
        Ok(Self { attention, attention_norm, cross, feed_forward })
    }

    pub fn cross_memory(&self, encoder: &Tensor) -> candle_core::Result<KeyValue> {
        let Some((attention, _)) = &self.cross else {
            candle_core::bail!("a T5 encoder block has no cross-attention memory");
        };
        attention.key_value(encoder, Route::Base)
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        bias: &Tensor,
        mask: Option<&Tensor>,
        encoder: Option<(&KeyValue, Option<&Tensor>)>,
        cache: Option<&mut KvCache>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let normed = self.attention_norm.forward(hidden, mode.pass)?;
        let attended = self.attention.self_attention(&normed, bias, mask, cache, mode)?;
        let mut hidden = clamp_half((hidden + attended)?)?;
        match (&self.cross, encoder) {
            (Some((attention, norm)), Some((memory, source_mask))) => {
                let normed = norm.forward(&hidden, mode.pass)?;
                let attended = attention.cross_attention(&normed, memory, source_mask, mode)?;
                hidden = clamp_half((hidden + attended)?)?;
            }
            (None, None) => {}
            (Some(_), None) => {
                candle_core::bail!("T5 decoder cross-attention requires encoder memory");
            }
            (None, Some(_)) => {
                candle_core::bail!("T5 encoder self-attention cannot consume decoder cross-memory");
            }
        }
        let forwarded = self.feed_forward.forward(&hidden, mode)?;
        clamp_half((hidden + forwarded)?)
    }
}

impl FeedForward {
    fn load(
        layer: VarBuilder<'_>,
        config: &Config,
        site: Option<(usize, &Adapters)>,
    ) -> candle_core::Result<Self> {
        let builder = layer.pp("DenseReluDense");
        let adapter = |target| {
            site.and_then(|(layer, adapters)| adapters.get(layer, target)).cloned()
        };
        let gate = if config.feed_forward.gated {
            Some(linear_no_bias(config.hidden, config.intermediate, builder.pp("wi_0"))?)
        } else {
            None
        };
        let up_name = if config.feed_forward.gated { "wi_1" } else { "wi" };
        Ok(Self {
            norm: config.norm().load(config.hidden, layer.pp("layer_norm"))?,
            gate,
            up: linear_no_bias(config.hidden, config.intermediate, builder.pp(up_name))?,
            down: linear_no_bias(config.intermediate, config.hidden, builder.pp("wo"))?,
            gate_adapter: adapter(Target::Gate),
            up_adapter: adapter(Target::Up),
            down_adapter: adapter(Target::Down),
            activation: config.feed_forward.activation,
        })
    }

    fn forward(&self, hidden: &Tensor, mode: Mode) -> candle_core::Result<Tensor> {
        let normed = self.norm.forward(hidden, mode.pass)?;
        let up = project(&self.up, self.up_adapter.as_ref(), &normed, mode.route)?;
        let activated = match &self.gate {
            Some(gate) => {
                let gate = project(gate, self.gate_adapter.as_ref(), &normed, mode.route)?;
                self.activation.gated(&gate, &up)?
            }
            None => self.activation.apply(&up)?,
        };
        project(&self.down, self.down_adapter.as_ref(), &activated, mode.route)
    }
}

/// The reference T5 block clamps every residual sublayer only if F16 overflow
/// occurred. F32/BF16 passes and finite F16 values are returned unchanged.
fn clamp_half(hidden: Tensor) -> candle_core::Result<Tensor> {
    if hidden.dtype() != DType::F16 {
        return Ok(hidden);
    }
    let has_infinity = hidden.abs()?.eq(f64::INFINITY)?.max_all()?.to_scalar::<u8>()?;
    // The reference asks whether any element is infinite, not whether a mean is finite.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    if has_infinity == 0 {
        return Ok(hidden);
    }
    // T5Block uses this published margin below the dtype's largest finite value.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    let bound = half::f16::MAX.to_f64() - 1000.0;
    hidden.clamp(-bound, bound)
}
