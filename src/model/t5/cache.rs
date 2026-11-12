//! Encoder memory is immutable; only decoder self-attention grows during generation.

use std::num::NonZeroUsize;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::kv_cache::KvCache;

use crate::failures::model::t5::{CACHE_EMPTY, CACHE_UNEVEN, inconsistent};

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
        let Some(first) = layers.first() else {
            return Err(inconsistent(&CACHE_EMPTY, "decoder", "a key-value cache of no layers"));
        };
        let position = first.current_seq_len();
        for (index, layer) in layers.iter().enumerate() {
            if layer.current_seq_len() != position {
                return Err(inconsistent(
                    &CACHE_UNEVEN,
                    "decoder",
                    &format!(
                        "cache layer {index} holds {} positions and the first {position}: an earlier pass ended half done and its cache was reused",
                        layer.current_seq_len()
                    ),
                ));
            }
        }
        Ok(position)
    }
}
