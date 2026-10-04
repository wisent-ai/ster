//! What a decode carries between steps: the rotary tables, the causal masks
//! it has already built, and the keys and values of the tokens behind it.

use std::{collections::HashMap, f32::consts::PI};

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize, Option<usize>), Tensor>,
    pub(super) use_kv_cache: bool,
    pub(super) kvs: Vec<Option<(Tensor, Tensor)>>,
    /// Rotary tables, held in F32 whatever the weights are. See [`Cache::new`].
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
    /// The dtype the base weights were mapped at. [`apply_rotary`] casts each
    /// rotated query and key back to it, so a half-precision checkpoint keeps
    /// a half-precision key-value cache and residual stream.
    pub(super) weights: DType,
    pub(super) device: Device,
}

impl Cache {
    /// `dtype` is the dtype the base weights were mapped at, and it is
    /// deliberately *not* the dtype of the rotary tables. Position angles are
    /// held in F32 whatever the weights are: `cos` and `sin` are indexed by
    /// absolute position, and in F16 the mantissa runs out long before
    /// `max_position_embeddings` does — two neighbouring positions late in the
    /// context round to the same angle, which rotates two different tokens
    /// identically and is invisible in the loss. The tables are one
    /// `[positions, head_dim / 2]` matrix, so holding them wide costs a few
    /// megabytes once rather than per forward, and [`apply_rotary`] casts each
    /// query and key back to the weights' dtype afterwards so the key-value
    /// cache still stores half-precision keys.
    ///
    /// `head_dim` is the architecture's head width, which a config may state
    /// apart from `hidden_size / num_attention_heads`; the rotation spans it.
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq = rotary_frequencies(config, head_dim);
        let theta = Tensor::new(inv_freq, device)?;
        let positions = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let angles = positions.matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos: angles.cos()?,
            sin: angles.sin()?,
            weights: dtype,
            device: device.clone(),
        })
    }

    /// The causal mask for `seq_len` queries starting at `index_pos`, with keys
    /// beyond `window` positions behind a query hidden too when the layer
    /// attends through a sliding window.
    pub(super) fn mask(
        &mut self,
        seq_len: usize,
        index_pos: usize,
        window: Option<usize>,
    ) -> candle_core::Result<Tensor> {
        let key = (seq_len, index_pos + seq_len, window);
        if let Some(mask) = self.masks.get(&key) {
            return Ok(mask.clone());
        }
        let key_len = index_pos + seq_len;
        let mut values = vec![0u8; seq_len * key_len];
        for query in 0..seq_len {
            let absolute_query = index_pos + query;
            for key_position in 0..key_len {
                if hidden_key(absolute_query, key_position, window) {
                    values[query * key_len + key_position] = 1;
                }
            }
        }
        let mask = Tensor::from_vec(values, (seq_len, key_len), &self.device)?;
        self.masks.insert(key, mask.clone());
        Ok(mask)
    }
}

/// Whether `query` may not see `key`: it is in the query's future, or it is
/// `window` or more positions behind it.
pub(super) fn hidden_key(query: usize, key: usize, window: Option<usize>) -> bool {
    key > query || window.is_some_and(|window| query - key >= window)
}

fn rotary_frequencies(config: &Config, head_dim: usize) -> Vec<f32> {
    let base: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|index| 1f32 / config.rope_theta.powf(index as f32 / head_dim as f32))
        .collect();
    match &config.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => base,
        Some(scaling) => {
            let low_wavelength =
                scaling.original_max_position_embeddings as f32 / scaling.low_freq_factor;
            let high_wavelength =
                scaling.original_max_position_embeddings as f32 / scaling.high_freq_factor;
            base.into_iter()
                .map(|frequency| {
                    let wavelength = 2.0 * PI / frequency;
                    if wavelength < high_wavelength {
                        frequency
                    } else if wavelength > low_wavelength {
                        frequency / scaling.factor
                    } else {
                        let smooth = (scaling.original_max_position_embeddings as f32 / wavelength
                            - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1.0 - smooth) * frequency / scaling.factor + smooth * frequency
                    }
                })
                .collect()
        }
    }
}
