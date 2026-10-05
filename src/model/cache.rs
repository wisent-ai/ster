//! What a decode carries between steps: the rotary frequencies, the causal
//! masks it has already built, and the keys and values of the tokens behind
//! it.

use std::{collections::HashMap, f32::consts::PI};

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

use super::RopeScaling;

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize, Option<usize>), Tensor>,
    pub(super) use_kv_cache: bool,
    /// Each attention layer's cached keys and values.
    pub(super) kvs: Vec<Option<(Tensor, Tensor)>>,
    /// Each recurrent mixer's decode state: its convolution history and its
    /// scan (Mamba, Mamba-2, LFM2's convolution). Kept apart from `kvs` so a
    /// layer that runs attention and a scan side by side (Falcon-H1) keeps
    /// both.
    pub(super) states: Vec<Option<(Tensor, Tensor)>>,
    /// Inkling's residual short convolutions' last `kernel - 1` inputs,
    /// `[batch, kernel - 1, channels]` in F32, under (layer, slot): a layer
    /// runs four of them.
    pub(super) convolutions: HashMap<(usize, usize), Tensor>,
    /// The keys and values of the current call that Gemma 4's
    /// key-value-sharing layers reuse, under the layer that produced them.
    /// Rewritten on every call, whether or not `kvs` keeps a history.
    pub(super) shared: Vec<Option<(Tensor, Tensor)>>,
    /// DeepSeek Sparse Attention's indexer keys per layer, `[batch, 1,
    /// keys, index_head_dim]` in F32, history included when the cache keeps
    /// one.
    pub(super) index_keys: Vec<Option<Tensor>>,
    /// The keys the last indexed layer of this call hid, which GLM-5's
    /// `shared` layers reuse.
    pub(super) index_mask: Option<Tensor>,
    /// LongCat-Flash's shortcut experts' output for the stored layer this
    /// call is in, set by its first half and taken by its second.
    pub(super) shortcut: Option<Tensor>,
    /// The last `emb_neighbor_num - 1` tokens of every sequence, which
    /// LongCat's n-gram embeddings hash the next call's tokens with.
    pub(super) ngram_context: Option<Vec<Vec<u32>>>,
    /// How many earlier chunks a windowed layer's query also sees when the
    /// model attends by chunks (Llama 4, Rnj-1); `None` for sliding
    /// windows.
    pub(super) chunk_lookback: Option<usize>,
    /// The global rotation, held in F32 whatever the weights are. See
    /// [`Cache::new`].
    pub(super) global: RotaryTable,
    /// The rotation sliding-window layers use, when the family gives them
    /// their own base (Gemma 3's `rope_local_base_freq`).
    pub(super) local: Option<RotaryTable>,
    /// LongRoPE's rotation for a sequence that runs past the original
    /// context, and that context's length.
    pub(super) long: Option<(RotaryTable, usize)>,
    /// The dtype the base weights were mapped at. [`apply_rotary`] casts each
    /// rotated query and key back to it, so a half-precision checkpoint keeps
    /// a half-precision key-value cache and residual stream.
    pub(super) weights: DType,
    pub(super) device: Device,
}

impl Cache {
    /// `dtype` is the dtype the base weights were mapped at, and it is
    /// deliberately *not* the dtype of the rotation. Position angles are
    /// computed in F32 whatever the weights are: in F16 the mantissa runs out
    /// long before `max_position_embeddings` does — two neighbouring
    /// positions late in the context round to the same angle, which rotates
    /// two different tokens identically and is invisible in the loss.
    /// [`apply_rotary`] casts each query and key back to the weights' dtype
    /// afterwards so the key-value cache still stores half-precision keys.
    ///
    /// Each rotation keeps only its `rotary_dim / 2` frequencies; the angles
    /// of the positions one call rotates are computed for that call, so the
    /// cache does not grow with `max_position_embeddings` (ten million for
    /// MiniMax-Text-01).
    ///
    /// The architecture supplies the width the global rotation spans
    /// (`rotary_dim`, which a config may set apart from the head width, or
    /// Gemma 4's full-attention head width), the scaling on the global
    /// rotation, and a second base for sliding-window layers when the family
    /// has one, over `rotary_dim`.
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &Config,
        architecture: &super::Architecture,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = architecture.global_rotary_dim();
        let global = rotary_frequencies(config, rotary_dim, config.rope_theta);
        let (global, long) = match &architecture.rope_scaling {
            RopeScaling::None => (RotaryTable::new(global, 1.0, device)?, None),
            RopeScaling::Linear(factor) => (
                RotaryTable::new(
                    global.into_iter().map(|frequency| frequency / factor).collect(),
                    1.0,
                    device,
                )?,
                None,
            ),
            RopeScaling::LongRope {
                short,
                long,
                original,
                short_attention,
                long_attention,
            } => {
                let rescaled = |factors: &[f32]| -> Vec<f32> {
                    global.iter().zip(factors).map(|(frequency, factor)| frequency / factor).collect()
                };
                let short = RotaryTable::new(rescaled(short), *short_attention, device)?;
                let long = RotaryTable::new(rescaled(long), *long_attention, device)?;
                (short, Some((long, *original)))
            }
            RopeScaling::Yarn {
                factor,
                original,
                beta_fast,
                beta_slow,
                attention,
                truncate,
            } => (
                RotaryTable::new(
                    yarn_frequencies(
                        &global,
                        rotary_dim,
                        config.rope_theta,
                        *factor,
                        *original,
                        (*beta_fast, *beta_slow),
                        *truncate,
                    ),
                    *attention,
                    device,
                )?,
                None,
            ),
            RopeScaling::Proportional { rotated, factor } => (
                RotaryTable::new(
                    global
                        .into_iter()
                        .enumerate()
                        .map(|(pair, frequency)| if pair < *rotated { frequency / factor } else { 0.0 })
                        .collect(),
                    1.0,
                    device,
                )?,
                None,
            ),
        };
        let local = match architecture.local_rope_theta {
            Some(theta) => Some(RotaryTable::new(
                base_frequencies(architecture.rotary_dim, theta),
                1.0,
                device,
            )?),
            None => None,
        };
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            states: vec![None; config.num_hidden_layers],
            convolutions: HashMap::new(),
            shared: vec![None; config.num_hidden_layers],
            index_keys: vec![None; config.num_hidden_layers],
            index_mask: None,
            shortcut: None,
            ngram_context: None,
            chunk_lookback: architecture.chunk_lookback,
            global,
            local,
            long,
            weights: dtype,
            device: device.clone(),
        })
    }

    /// The causal mask for `seq_len` queries starting at `index_pos`, with keys
    /// beyond `window` positions behind a query hidden too when the layer
    /// attends through a sliding window, or outside its chunks when the
    /// model attends by chunks.
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
                if hidden_key(absolute_query, key_position, window, self.chunk_lookback) {
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
/// `window` or more positions behind it — or, when the model attends by
/// chunks (`chunk_lookback`), `window` is the chunk size and `key` lies in
/// a chunk more than `chunk_lookback` chunks before the query's (Llama 4's
/// chunked attention looks back none, Rnj-1's one).
pub(super) fn hidden_key(query: usize, key: usize, window: Option<usize>, chunk_lookback: Option<usize>) -> bool {
    if key > query {
        return true;
    }
    match (window, chunk_lookback) {
        (Some(size), Some(lookback)) if size > 0 => key / size + lookback < query / size,
        (Some(window), None) => query - key >= window,
        _ => false,
    }
}

/// One rotation: its frequencies, `[1, rotary_dim / 2]` in F32, and the
/// magnitude its `cos` and `sin` are multiplied by (LongRoPE's and YaRN's
/// attention factor; one otherwise, where the multiply is skipped).
#[derive(Debug, Clone)]
pub(super) struct RotaryTable {
    frequencies: Tensor,
    magnitude: f64,
}

impl RotaryTable {
    fn new(frequencies: Vec<f32>, magnitude: f32, device: &Device) -> Result<Self> {
        let width = frequencies.len();
        Ok(Self {
            frequencies: Tensor::from_vec(frequencies, (1, width), device)?,
            magnitude: f64::from(magnitude),
        })
    }

    /// `cos` and `sin` of positions `start .. start + count` times every
    /// frequency, `[count, rotary_dim / 2]`, in F32.
    pub(super) fn angles(&self, start: usize, count: usize) -> Result<(Tensor, Tensor)> {
        let device = self.frequencies.device();
        let positions = Tensor::arange(start as u32, (start + count) as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((count, 1))?;
        let angles = positions.matmul(&self.frequencies)?;
        if self.magnitude == 1.0 {
            return Ok((angles.cos()?, angles.sin()?));
        }
        Ok(((angles.cos()? * self.magnitude)?, (angles.sin()? * self.magnitude)?))
    }
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
/// bound a linear ramp — rounded out to whole pairs when `truncate` holds —;
/// above the band a frequency keeps its value (extrapolation), below it is
/// divided by `factor` (interpolation), and in the band the two are blended
/// along the ramp.
fn yarn_frequencies(
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
    let (low, high) = if truncate { (low.floor(), high.ceil()) } else { (low, high) };
    let low = low.max(0.0);
    let high = high.min(rotary_dim as f32 - 1.0);
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
