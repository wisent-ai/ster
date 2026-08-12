use std::{collections::{BTreeMap, HashMap}, f32::consts::PI};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

#[derive(Debug, Clone)]
pub struct SteeringPlan {
    vectors: BTreeMap<usize, Tensor>,
    strength: f64,
    hidden_size: usize,
}

impl SteeringPlan {
    pub fn new(
        vectors: impl IntoIterator<Item = (usize, Vec<f32>)>,
        strength: f64,
        hidden_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (layer, values) in vectors {
            if values.len() != hidden_size {
                bail!(
                    "layer {layer} steering vector width {} does not match model width {hidden_size}",
                    values.len()
                );
            }
            let tensor = Tensor::from_vec(values, hidden_size, device)?.to_dtype(dtype)?;
            tensors.insert(layer, tensor);
        }
        if tensors.is_empty() {
            bail!("steering plan contains no vectors");
        }
        Ok(Self { vectors: tensors, strength, hidden_size })
    }

    fn vector(&self, layer: usize) -> Option<&Tensor> {
        self.vectors.get(&layer)
    }
}

#[derive(Debug)]
pub struct ForwardOutput {
    pub logits: Tensor,
    pub activations: BTreeMap<usize, Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let inv_freq = rotary_frequencies(config);
        let theta = Tensor::new(inv_freq, device)?;
        let positions = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let angles = positions.matmul(&theta.reshape((1, theta.elem_count()))?)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos: angles.cos()?.to_dtype(dtype)?,
            sin: angles.sin()?.to_dtype(dtype)?,
            device: device.clone(),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> candle_core::Result<Tensor> {
        let key = (seq_len, index_pos + seq_len);
        if let Some(mask) = self.masks.get(&key) {
            return Ok(mask.clone());
        }
        let key_len = index_pos + seq_len;
        let mut values = vec![0u8; seq_len * key_len];
        for query in 0..seq_len {
            let absolute_query = index_pos + query;
            for key_position in (absolute_query + 1)..key_len {
                values[query * key_len + key_position] = 1;
            }
        }
        let mask = Tensor::from_vec(values, (seq_len, key_len), &self.device)?;
        self.masks.insert(key, mask.clone());
        Ok(mask)
    }
}

fn rotary_frequencies(config: &Config) -> Vec<f32> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let base: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|index| 1f32 / config.rope_theta.powf(index as f32 / head_dim as f32))
        .collect();
    match &config.rope_scaling {
        None | Some(Llama3RopeConfig { rope_type: Llama3RopeType::Default, .. }) => base,
        Some(scaling) => {
            let low_wavelength = scaling.original_max_position_embeddings as f32 / scaling.low_freq_factor;
            let high_wavelength = scaling.original_max_position_embeddings as f32 / scaling.high_freq_factor;
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

#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(builder: VarBuilder<'_>, config: &Config) -> candle_core::Result<Self> {
        let input = config.hidden_size;
        let query_width = config.hidden_size;
        let key_value_width = config.hidden_size / config.num_attention_heads * config.num_key_value_heads;
        Ok(Self {
            query: linear_no_bias(input, query_width, builder.pp("q_proj"))?,
            key: linear_no_bias(input, key_value_width, builder.pp("k_proj"))?,
            value: linear_no_bias(input, key_value_width, builder.pp("v_proj"))?,
            output: linear_no_bias(query_width, input, builder.pp("o_proj"))?,
            heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, hidden_size) = hidden.dims3()?;
        let query = self.query.forward(hidden)?
            .reshape((batch, sequence, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut key = self.key.forward(hidden)?
            .reshape((batch, sequence, self.key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut value = self.value.forward(hidden)?
            .reshape((batch, sequence, self.key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query = apply_rotary(&query, index_pos, &cache.cos, &cache.sin)?;
        key = apply_rotary(&key, index_pos, &cache.cos, &cache.sin)?;
        if cache.use_kv_cache {
            if let Some((cached_key, cached_value)) = &cache.kvs[layer] {
                key = Tensor::cat(&[cached_key, &key], 2)?.contiguous()?;
                value = Tensor::cat(&[cached_value, &value], 2)?.contiguous()?;
            }
            cache.kvs[layer] = Some((key.clone(), value.clone()));
        }
        let repeats = self.heads / self.key_value_heads;
        let key = repeat_key_value(key, repeats)?;
        let value = repeat_key_value(value, repeats)?;
        let input_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;
        let attention = (query.matmul(&key.t()?)? / (self.head_dim as f64).sqrt())?;
        let attention = if sequence == 1 {
            attention
        } else {
            let mask = cache.mask(sequence, index_pos)?.broadcast_as(attention.shape())?;
            masked_fill(&attention, &mask, f32::NEG_INFINITY)?
        };
        let attention = candle_nn::ops::softmax_last_dim(&attention)?;
        let output = attention.matmul(&value.contiguous()?)?.to_dtype(input_dtype)?;
        let output = output.transpose(1, 2)?.reshape((batch, sequence, hidden_size))?;
        self.output.forward(&output)
    }
}

fn apply_rotary(input: &Tensor, index_pos: usize, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, sequence, _) = input.dims4()?;
    let cos = cos.narrow(0, index_pos, sequence)?;
    let sin = sin.narrow(0, index_pos, sequence)?;
    candle_nn::rotary_emb::rope(input, &cos, &sin)
}

fn repeat_key_value(input: Tensor, repeats: usize) -> candle_core::Result<Tensor> {
    if repeats == 1 {
        return Ok(input);
    }
    let (batch, key_value_heads, sequence, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((batch, key_value_heads, repeats, sequence, head_dim))?
        .reshape((batch, key_value_heads * repeats, sequence, head_dim))
}

fn masked_fill(values: &Tensor, mask: &Tensor, replacement: f32) -> candle_core::Result<Tensor> {
    let replacement = Tensor::new(replacement, values.device())?.broadcast_as(mask.shape())?;
    mask.where_cond(&replacement, values)
}

#[derive(Debug, Clone)]
struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(builder: VarBuilder<'_>, config: &Config) -> candle_core::Result<Self> {
        Ok(Self {
            gate: linear_no_bias(config.hidden_size, config.intermediate_size, builder.pp("gate_proj"))?,
            up: linear_no_bias(config.hidden_size, config.intermediate_size, builder.pp("up_proj"))?,
            down: linear_no_bias(config.intermediate_size, config.hidden_size, builder.pp("down_proj"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.gate.forward(hidden)?)? * self.up.forward(hidden)?)?;
        self.down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    attention_norm: RmsNorm,
    attention: Attention,
    feed_forward_norm: RmsNorm,
    feed_forward: FeedForward,
}

impl DecoderLayer {
    fn load(builder: VarBuilder<'_>, config: &Config) -> candle_core::Result<Self> {
        Ok(Self {
            attention_norm: rms_norm(config.hidden_size, config.rms_norm_eps, builder.pp("input_layernorm"))?,
            attention: Attention::load(builder.pp("self_attn"), config)?,
            feed_forward_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("post_attention_layernorm"),
            )?,
            feed_forward: FeedForward::load(builder.pp("mlp"), config)?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let attention = self.attention.forward(
            &self.attention_norm.forward(hidden)?,
            index_pos,
            layer,
            cache,
        )?;
        let hidden = (hidden + attention)?;
        let feed_forward = self.feed_forward.forward(&self.feed_forward_norm.forward(&hidden)?)?;
        hidden + feed_forward
    }
}

#[derive(Debug, Clone)]
pub struct SteeringLlama {
    embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    config: Config,
}

impl SteeringLlama {
    pub fn load(builder: VarBuilder<'_>, config: Config) -> Result<Self> {
        let embeddings = embedding(config.vocab_size, config.hidden_size, builder.pp("model.embed_tokens"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, builder.pp("lm_head"))?
        };
        let final_norm = rms_norm(config.hidden_size, config.rms_norm_eps, builder.pp("model.norm"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| DecoderLayer::load(builder.pp(format!("model.layers.{index}")), &config))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self { embeddings, layers, final_norm, lm_head, config })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn forward(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
    ) -> Result<ForwardOutput> {
        let (_, sequence) = tokens.dims2()?;
        let mut hidden = self.embeddings.forward(tokens)?;
        let mut activations = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, index_pos, index, cache)?;
            if capture_layers.binary_search(&index).is_ok() {
                let activation = hidden
                    .i((0, sequence - 1, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                activations.insert(index, activation);
            }
            if let Some(plan) = steering {
                if let Some(vector) = plan.vector(index) {
                    let scaled = (vector * plan.strength)?
                        .reshape((1, 1, plan.hidden_size))?;
                    hidden = hidden.broadcast_add(&scaled)
                        .with_context(|| format!("failed to apply steering at layer {index}"))?;
                }
            }
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let last = hidden.i((.., sequence - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&last)?.to_dtype(DType::F32)?;
        Ok(ForwardOutput { logits, activations })
    }
}
