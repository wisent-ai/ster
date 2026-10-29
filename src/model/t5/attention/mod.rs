//! T5 attention has learned position bias and no query/key dot-product scale.

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, VarBuilder, kv_cache::KvCache, linear_no_bias};

use crate::{
    lora::{Adapter, Adapters, Target},
    model::{
        Mode, Pass, Route,
        attention::{masked_fill, project},
    },
};

use super::config::Config;

pub(super) mod mask;
pub(super) mod relative;

#[derive(Debug, Clone)]
pub(super) struct KeyValue {
    pub key: Tensor,
    pub value: Tensor,
}

#[derive(Debug, Clone)]
pub(super) struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_adapter: Option<Adapter>,
    key_adapter: Option<Adapter>,
    value_adapter: Option<Adapter>,
    output_adapter: Option<Adapter>,
    heads: usize,
    head_width: usize,
    attention_width: usize,
}

impl Attention {
    /// Only decoder self-attention supplies an adapter site. Encoder and
    /// cross-attention projections read the same frozen checkpoint throughout.
    pub fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        site: Option<(usize, &Adapters)>,
    ) -> candle_core::Result<Self> {
        let adapter = |target| {
            site.and_then(|(layer, adapters)| adapters.get(layer, target))
                .cloned()
        };
        Ok(Self {
            query: linear_no_bias(config.hidden, config.attention_width, builder.pp("q"))?,
            key: linear_no_bias(config.hidden, config.attention_width, builder.pp("k"))?,
            value: linear_no_bias(config.hidden, config.attention_width, builder.pp("v"))?,
            output: linear_no_bias(config.attention_width, config.hidden, builder.pp("o"))?,
            query_adapter: adapter(Target::Query),
            key_adapter: adapter(Target::Key),
            value_adapter: adapter(Target::Value),
            output_adapter: adapter(Target::Output),
            heads: config.heads,
            head_width: config.head_width,
            attention_width: config.attention_width,
        })
    }

    /// Cross-attention calls this once for its encoder memory, then keeps the result.
    pub fn key_value(&self, hidden: &Tensor, route: Route) -> candle_core::Result<KeyValue> {
        Ok(KeyValue {
            key: self.heads(project(
                &self.key,
                self.key_adapter.as_ref(),
                hidden,
                route,
            )?)?,
            value: self.heads(project(
                &self.value,
                self.value_adapter.as_ref(),
                hidden,
                route,
            )?)?,
        })
    }

    pub fn self_attention(
        &self,
        hidden: &Tensor,
        bias: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<&mut KvCache>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        if cache.is_some() && mode.pass == Pass::Differentiable {
            candle_core::bail!("a differentiable T5 pass cannot mutate a key-value cache");
        }
        let current = self.key_value(hidden, mode.route)?;
        let memory = match cache {
            Some(cache) => {
                let (key, value) = cache.append(&current.key, &current.value)?;
                KeyValue { key, value }
            }
            None => current,
        };
        self.attend(hidden, &memory, Some(bias), mask, mode)
    }

    /// Encoder memory is bidirectional and never receives a decoder causal mask.
    pub fn cross_attention(
        &self,
        hidden: &Tensor,
        memory: &KeyValue,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        self.attend(hidden, memory, None, mask, mode)
    }

    fn heads(&self, projected: Tensor) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = projected.dims3()?;
        projected
            .reshape((batch, sequence, self.heads, self.head_width))?
            // [batch, sequence, heads, width] -> [batch, heads, sequence, width].
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            .transpose(1, 2)?
            .contiguous()
    }

    fn attend(
        &self,
        hidden: &Tensor,
        memory: &KeyValue,
        bias: Option<&Tensor>,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (batch, queries, _) = hidden.dims3()?;
        let query = self
            .heads(project(
                &self.query,
                self.query_adapter.as_ref(),
                hidden,
                mode.route,
            )?)?
            .to_dtype(DType::F32)?;
        // T5 uses q @ k.T without the head-width scaling other families apply.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        let key = memory.key.to_dtype(DType::F32)?.transpose(2, 3)?;
        let mut scores = query.matmul(&key)?;
        if let Some(bias) = bias {
            scores = scores.broadcast_add(bias)?;
        }
        if let Some(mask) = mask {
            scores = masked_fill(
                &scores,
                &mask.broadcast_as(scores.shape())?,
                f32::NEG_INFINITY,
            )?;
        }
        let probabilities = match mode.pass {
            Pass::Inference => candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?,
            Pass::Differentiable => candle_nn::ops::softmax(&scores, D::Minus1)?,
        };
        let context = probabilities
            .matmul(&memory.value.to_dtype(DType::F32)?)?
            // Restore sequence-major residual layout before joining the heads.
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, queries, self.attention_width))?
            .to_dtype(hidden.dtype())?;
        project(
            &self.output,
            self.output_adapter.as_ref(),
            &context,
            mode.route,
        )
    }
}
