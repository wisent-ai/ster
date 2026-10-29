//! Encoder memory is immutable; only decoder self-attention grows during generation.

use std::num::NonZeroUsize;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::kv_cache::KvCache;

use super::attention::KeyValue;

#[derive(Debug, Clone)]
pub(super) struct EncoderState {
    pub hidden: Tensor,
    pub cross_attention: Vec<KeyValue>,
    pub mask: Option<Tensor>,
}

#[derive(Debug)]
pub(crate) struct Cache {
    pub(super) encoder: Option<EncoderState>,
    pub(super) decoder: Option<Vec<KvCache>>,
}

impl Cache {
    /// Capacity comes from the caller's actual decoder prefix and requested output
    /// (or longest scoring suffix). No cache allocation is made for a training pass.
    pub fn new(layers: usize, capacity: Option<NonZeroUsize>) -> Self {
        let decoder = capacity.map(|capacity| {
            // Each decoder block retains its own sequence-axis key/value storage.
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            (0..layers)
                // Cached tensors are [batch, heads, sequence, head_width].
                // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
                .map(|_| KvCache::new(2, capacity.get()))
                .collect()
        });
        Self {
            encoder: None,
            decoder,
        }
    }

    pub fn position(&self) -> Result<usize> {
        let Some(layers) = &self.decoder else {
            // A pass without decoder history starts at the sequence origin.
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            return Ok(0);
        };
        let first = layers
            .first()
            .context("T5 decoder cache contains no layers")?;
        let position = first.current_seq_len();
        for (index, layer) in layers.iter().enumerate() {
            if layer.current_seq_len() != position {
                bail!(
                    "T5 decoder cache layer {index} holds {} positions, but the first layer holds {position}; discard the incomplete pass before reusing its cache",
                    layer.current_seq_len()
                );
            }
        }
        Ok(position)
    }
}
