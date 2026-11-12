//! The rotary tables a cache carries: the frequencies a rotation turns at,
//! from the plain base, Llama 3's smoothed scaling, or YaRN's ramp.

use std::f32::consts::PI;

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

/// One rotation: its frequencies, `[1, rotary_dim / 2]` in F32, and the
/// magnitude its `cos` and `sin` are multiplied by (LongRoPE's and YaRN's
/// attention factor; one otherwise, where the multiply is skipped).
#[derive(Debug, Clone)]
pub(in crate::model) struct RotaryTable {
    frequencies: Tensor,
    magnitude: f64,
}

impl RotaryTable {
    pub(in crate::model) fn new(frequencies: Vec<f32>, magnitude: f32, device: &Device) -> Result<Self> {
        let width = frequencies.len();
        Ok(Self {
            frequencies: Tensor::from_vec(frequencies, (1, width), device)?,
            magnitude: f64::from(magnitude),
        })
    }

    /// `cos` and `sin` of positions `start .. start + count` times every
    /// frequency, `[count, rotary_dim / 2]`, in F32.
    pub(in crate::model) fn angles(&self, start: usize, count: usize) -> Result<(Tensor, Tensor)> {
        let device = self.frequencies.device();
        let positions = Tensor::arange(start as u32, (start + count) as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((count, 1))?;
        let angles = positions.matmul(&self.frequencies)?;
        if self.magnitude == 1.0 {
            return Ok((angles.cos()?, angles.sin()?));
        }
        Ok((
            (angles.cos()? * self.magnitude)?,
            (angles.sin()? * self.magnitude)?,
        ))
    }

    /// `cos` and `sin` of each of `positions` times every frequency,
    /// `[positions, rotary_dim / 2]`, in F32.
    pub(in crate::model) fn angles_at(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let device = self.frequencies.device();
        let count = positions.len();
        let positions: Vec<f32> = positions.iter().map(|position| *position as f32).collect();
        let angles = Tensor::from_vec(positions, (count, 1), device)?.matmul(&self.frequencies)?;
        if self.magnitude == 1.0 {
            return Ok((angles.cos()?, angles.sin()?));
        }
        Ok((
            (angles.cos()? * self.magnitude)?,
            (angles.sin()? * self.magnitude)?,
        ))
    }
}

pub(in crate::model) fn base_frequencies(head_dim: usize, theta: f32) -> Vec<f32> {
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
/// bound a linear ramp — rounded out to whole pairs when `truncate` holds —;
/// above the band a frequency keeps its value (extrapolation), below it is
/// divided by `factor` (interpolation), and in the band the two are blended
/// along the ramp.
pub(in crate::model) fn yarn_frequencies(
    base: &[f32],
    rotary_dim: usize,
    theta: f32,
    factor: f32,
    original: usize,
    (beta_fast, beta_slow): (f32, f32),
    truncate: bool,
) -> Vec<f32> {
    let correction = |turns: f32| -> f32 {
        rotary_dim as f32 * (original as f32 / (turns * 2.0 * PI)).ln() / (2.0 * theta.ln())
    };
    let (low, high) = (correction(beta_fast), correction(beta_slow));
    let (low, high) = if truncate {
        (low.floor(), high.ceil())
    } else {
        (low, high)
    };
    let low = low.max(0.0);
    let high = high.min(rotary_dim as f32 - 1.0);
    let high = if high == low {
        high + YARN_RAMP_WIDENING
    } else {
        high
    };
    base.iter()
        .enumerate()
        .map(|(index, frequency)| {
            let ramp = ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let extrapolation = 1.0 - ramp;
            frequency / factor * (1.0 - extrapolation) + frequency * extrapolation
        })
        .collect()
}

pub(super) fn rotary_frequencies(config: &Config, head_dim: usize, theta: f32) -> Vec<f32> {
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
