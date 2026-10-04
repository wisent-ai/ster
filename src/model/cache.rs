//! What a decode carries between steps: the rotary tables, the causal masks
//! it has already built, and the keys and values of the tokens behind it.

use std::{collections::HashMap, f32::consts::PI};

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

use super::RopeScaling;

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize, Option<usize>), Tensor>,
    pub(super) use_kv_cache: bool,
    pub(super) kvs: Vec<Option<(Tensor, Tensor)>>,
    /// Rotary tables, held in F32 whatever the weights are. See [`Cache::new`].
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
    /// The tables sliding-window layers rotate with, when the family gives
    /// them their own base (Gemma 3's `rope_local_base_freq`).
    pub(super) local: Option<(Tensor, Tensor)>,
    /// LongRoPE's tables for a sequence that runs past the original context,
    /// and that context's length.
    pub(super) long: Option<(Tensor, Tensor, usize)>,
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
    /// `[positions, rotary_dim / 2]` matrix, so holding them wide costs a few
    /// megabytes once rather than per forward, and [`apply_rotary`] casts each
    /// query and key back to the weights' dtype afterwards so the key-value
    /// cache still stores half-precision keys.
    ///
    /// The architecture supplies the width the rotation spans (`rotary_dim`,
    /// which a config may set apart from the head width), the scaling on the
    /// global table, and a second base for sliding-window layers when the
    /// family has one.
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        architecture: &super::Architecture,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = architecture.rotary_dim;
        let positions = config.max_position_embeddings;
        let global = rotary_frequencies(config, rotary_dim, config.rope_theta);
        let ((cos, sin), long) = match &architecture.rope_scaling {
            RopeScaling::None => (angle_tables(global, positions, 1.0, device)?, None),
            RopeScaling::Linear(factor) => (
                angle_tables(
                    global.into_iter().map(|frequency| frequency / factor).collect(),
                    positions,
                    1.0,
                    device,
                )?,
                None,
            ),
            RopeScaling::LongRope {
                short,
                long,
                original,
                attention,
            } => {
                let rescaled = |factors: &[f32]| -> Vec<f32> {
                    global.iter().zip(factors).map(|(frequency, factor)| frequency / factor).collect()
                };
                let short = angle_tables(rescaled(short), positions, *attention, device)?;
                let (long_cos, long_sin) =
                    angle_tables(rescaled(long), positions, *attention, device)?;
                (short, Some((long_cos, long_sin, *original)))
            }
            RopeScaling::Yarn {
                factor,
                original,
                beta_fast,
                beta_slow,
                attention,
            } => (
                angle_tables(
                    yarn_frequencies(
                        &global,
                        rotary_dim,
                        config.rope_theta,
                        *factor,
                        *original,
                        (*beta_fast, *beta_slow),
                    ),
                    positions,
                    *attention,
                    device,
                )?,
                None,
            ),
        };
        let local = match architecture.local_rope_theta {
            Some(theta) => Some(angle_tables(
                base_frequencies(rotary_dim, theta),
                positions,
                1.0,
                device,
            )?),
            None => None,
        };
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos,
            sin,
            local,
            long,
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

/// `cos` and `sin` of every position times every frequency, `[positions,
/// rotary_dim / 2]`, in F32, each multiplied by `magnitude` (LongRoPE's
/// attention factor; one otherwise, where the multiply is skipped).
fn angle_tables(
    frequencies: Vec<f32>,
    positions: usize,
    magnitude: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta = Tensor::new(frequencies, device)?;
    let positions = Tensor::arange(0, positions as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((positions, 1))?;
    let angles = positions.matmul(&theta.reshape((1, theta.elem_count()))?)?;
    if magnitude == 1.0 {
        return Ok((angles.cos()?, angles.sin()?));
    }
    let magnitude = f64::from(magnitude);
    Ok(((angles.cos()? * magnitude)?, (angles.sin()? * magnitude)?))
}

fn base_frequencies(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim)
        .step_by(2)
        .map(|index| 1f32 / theta.powf(index as f32 / head_dim as f32))
        .collect()
}

/// How far Transformers widens an empty YaRN band so its ramp does not
/// divide by zero.
const YARN_RAMP_WIDENING: f32 = 0.001;

/// YaRN's frequencies, as Transformers' `_compute_yarn_parameters` computes
/// them: the dimension pair where a frequency completes `beta_fast` turns
/// over `original` positions and the one where it completes `beta_slow`
/// bound a linear ramp; above the band a frequency keeps its value
/// (extrapolation), below it is divided by `factor` (interpolation), and in
/// the band the two are blended along the ramp.
fn yarn_frequencies(
    base: &[f32],
    rotary_dim: usize,
    theta: f32,
    factor: f32,
    original: usize,
    (beta_fast, beta_slow): (f32, f32),
) -> Vec<f32> {
    let correction = |turns: f32| -> f32 {
        rotary_dim as f32 * (original as f32 / (turns * 2.0 * PI)).ln() / (2.0 * theta.ln())
    };
    let low = correction(beta_fast).floor().max(0.0);
    let high = correction(beta_slow).ceil().min(rotary_dim as f32 - 1.0);
    let high = if high == low { high + YARN_RAMP_WIDENING } else { high };
    base.iter()
        .enumerate()
        .map(|(index, frequency)| {
            let ramp = ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let extrapolation = 1.0 - ramp;
            frequency / factor * (1.0 - extrapolation) + frequency * extrapolation
        })
        .collect()
}

fn rotary_frequencies(config: &Config, head_dim: usize, theta: f32) -> Vec<f32> {
    let base = base_frequencies(head_dim, theta);
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
