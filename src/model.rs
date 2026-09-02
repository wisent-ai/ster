use std::{collections::{BTreeMap, HashMap}, f32::consts::PI};

use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};

use crate::lora::{Adapter, Adapters, Target};

/// Whether the forward pass must be differentiable.
///
/// Ster's decode loop leans on three fused Candle kernels — `rotary_emb::rope`,
/// `ops::softmax_last_dim` and `ops::rms_norm` — and every one of them ends in
/// an `apply_op*_no_bwd` call, so none of them records a node the autograd tape
/// can walk back through. Training therefore selects composed equivalents at
/// exactly those three call sites. Nothing else in the decoder changes, and
/// inference never pays for the swap: it is chosen by the caller, never by
/// default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Inference,
    Differentiable,
}

/// Whether the adapters attached to this model take part in a forward pass.
///
/// Preference optimization scores every sequence twice: once under the policy
/// and once under the frozen reference it is not allowed to drift far from.
/// The reference is not a second checkpoint. It is these same read-only base
/// weights with the low-rank update left out, because `B` starts at zero and
/// the adapters are the only tensors training ever changes — so skipping them
/// for one pass reproduces the reference distribution exactly, at the cost of
/// one enum comparison per projection instead of a second multi-gigabyte mmap.
/// That is the whole reason adapters are attached to the decoder rather than
/// folded into the projection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Adapted,
    Base,
}

/// How much of the vocabulary projection the caller actually needs.
///
/// Decoding samples one token, so it projects the final position and leaves
/// the rest of the `[sequence, vocab]` matmul undone. Anything that scores a
/// whole sequence — a training loss, a reference log-probability, a held-out
/// perplexity — needs every position. A reward head needs none of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    LastPosition,
    EveryPosition,
    /// No vocabulary projection at all.
    ///
    /// A reward head maps the residual stream to one scalar and never looks at
    /// a token distribution, so projecting `[sequence, hidden]` onto a
    /// vocabulary of a hundred thousand columns would compute — and, while
    /// training, backpropagate — the widest matmul in the pass only to drop it.
    Hidden,
}

/// The three independent choices one forward pass makes.
///
/// They used to be one. [`Pass`] picked the kernels *and* the readout, which
/// worked while the only differentiable caller wanted every position and the
/// only inference caller wanted the last. Scoring a sequence under a model
/// nobody is training — a preference reference, a held-out evaluation — wants
/// the fused kernels and every position at once, and that pairing had no way
/// to say so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode {
    pub pass: Pass,
    pub route: Route,
    pub readout: Readout,
}

impl Mode {
    /// Autoregressive decoding: fused kernels, adapters on, one row of logits.
    pub const DECODE: Self = Self {
        pass: Pass::Inference,
        route: Route::Adapted,
        readout: Readout::LastPosition,
    };

    /// A training step: composed kernels so the tape can be walked back, and
    /// logits at every position because the loss scores every position.
    pub const TRAIN: Self = Self {
        pass: Pass::Differentiable,
        route: Route::Adapted,
        readout: Readout::EveryPosition,
    };

    /// Scoring a whole sequence with no gradient. The fused kernels are the
    /// point: nothing here is backpropagated, so paying for the composed forms
    /// would buy an autograd tape that is thrown away.
    pub const fn score(route: Route) -> Self {
        Self { pass: Pass::Inference, route, readout: Readout::EveryPosition }
    }

    /// A reward model's forward: composed kernels, adapters on, and the
    /// residual stream instead of a token distribution.
    pub const REWARD: Self = Self {
        pass: Pass::Differentiable,
        route: Route::Adapted,
        readout: Readout::Hidden,
    };

    /// A trained reward model judging text: fused kernels, its own adapters
    /// on, and no vocabulary. Nothing here is trained — the model doing the
    /// scoring in a policy-optimization loop is frozen by definition, or it
    /// would be moving the target it is being optimized against.
    pub const JUDGE: Self = Self {
        pass: Pass::Inference,
        route: Route::Adapted,
        readout: Readout::Hidden,
    };
}

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
    /// The vocabulary projection the readout asked for, or `None` when it
    /// asked for none.
    ///
    /// A reward model reads the residual stream and never touches the
    /// vocabulary; on a real checkpoint that projection is the widest matmul
    /// in the pass, so skipping it is worth an `Option` at the two call sites
    /// that unwrap one.
    pub logits: Option<Tensor>,
    /// The residual stream after the final norm, `[batch, sequence, hidden]`.
    ///
    /// Always returned, because a `Tensor` is a handle and returning it costs
    /// a refcount rather than a copy.
    pub hidden: Tensor,
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
    query_adapter: Option<Adapter>,
    key_adapter: Option<Adapter>,
    value_adapter: Option<Adapter>,
    output_adapter: Option<Adapter>,
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let input = config.hidden_size;
        let query_width = config.hidden_size;
        let key_value_width = config.hidden_size / config.num_attention_heads * config.num_key_value_heads;
        Ok(Self {
            query: linear_no_bias(input, query_width, builder.pp("q_proj"))?,
            key: linear_no_bias(input, key_value_width, builder.pp("k_proj"))?,
            value: linear_no_bias(input, key_value_width, builder.pp("v_proj"))?,
            output: linear_no_bias(query_width, input, builder.pp("o_proj"))?,
            query_adapter: adapters.get(layer, Target::Query).cloned(),
            key_adapter: adapters.get(layer, Target::Key).cloned(),
            value_adapter: adapters.get(layer, Target::Value).cloned(),
            output_adapter: adapters.get(layer, Target::Output).cloned(),
            heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        })
    }

    /// One attention block, optionally under a caller-supplied mask.
    ///
    /// `mask` is the combined causal and key-padding constraint a batched
    /// caller built once for the whole stack, shaped `[batch, 1, sequence,
    /// keys]` so the unit head axis broadcasts across every head. When it is
    /// `None` this is the single-sequence path and the mask comes from the
    /// cache, exactly as it always did.
    fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, hidden_size) = hidden.dims3()?;
        let query = project(&self.query, self.query_adapter.as_ref(), hidden, mode.route)?
            .reshape((batch, sequence, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut key = project(&self.key, self.key_adapter.as_ref(), hidden, mode.route)?
            .reshape((batch, sequence, self.key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut value = project(&self.value, self.value_adapter.as_ref(), hidden, mode.route)?
            .reshape((batch, sequence, self.key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query = apply_rotary(&query, index_pos, &cache.cos, &cache.sin, mode.pass)?;
        key = apply_rotary(&key, index_pos, &cache.cos, &cache.sin, mode.pass)?;
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
        let attention = match mask {
            // A supplied mask already carries the causal constraint, so it is
            // applied at every batch size instead of only when the query axis
            // is longer than one — a padded row must not attend to its own
            // filler however short the query axis happens to be.
            Some(mask) => {
                let mask = mask.broadcast_as(attention.shape())?;
                masked_fill(&attention, &mask, f32::NEG_INFINITY)?
            }
            // A lone query with no supplied mask can only reach keys that
            // already exist, so there is nothing causality would remove.
            None if sequence == 1 => attention,
            None => {
                let mask = cache.mask(sequence, index_pos)?.broadcast_as(attention.shape())?;
                masked_fill(&attention, &mask, f32::NEG_INFINITY)?
            }
        };
        // `softmax_last_dim` is `apply_op1_no_bwd` (candle-nn-0.11.0/src/ops.rs:438).
        // `ops::softmax` is the same softmax spelled out of `max_keepdim`,
        // `broadcast_sub`, `exp`, `sum_keepdim` and `broadcast_div`, all of which
        // record a backward node.
        let attention = match mode.pass {
            Pass::Inference => candle_nn::ops::softmax_last_dim(&attention)?,
            Pass::Differentiable => candle_nn::ops::softmax(&attention, candle_core::D::Minus1)?,
        };
        let output = attention.matmul(&value.contiguous()?)?.to_dtype(input_dtype)?;
        let output = output.transpose(1, 2)?.reshape((batch, sequence, hidden_size))?;
        project(&self.output, self.output_adapter.as_ref(), &output, mode.route)
    }
}

/// Applies a projection, adding the low-rank update when this site is adapted
/// and `route` asks for it.
///
/// The `None` arm is the historical code path down to the op: one nullable
/// check, no tensor allocated, no dtype touched. An unadapted model — the whole
/// steering product — therefore costs a null-pointer test per projection, and
/// a reference pass over an adapted model costs one enum comparison more.
fn project(
    base: &Linear,
    adapter: Option<&Adapter>,
    hidden: &Tensor,
    route: Route,
) -> candle_core::Result<Tensor> {
    let projected = base.forward(hidden)?;
    match adapter.filter(|_| route == Route::Adapted) {
        None => Ok(projected),
        Some(adapter) => projected + adapter.forward(hidden)?,
    }
}

fn apply_rotary(
    input: &Tensor,
    index_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    pass: Pass,
) -> candle_core::Result<Tensor> {
    let (_, _, sequence, _) = input.dims4()?;
    let cos = cos.narrow(0, index_pos, sequence)?;
    let sin = sin.narrow(0, index_pos, sequence)?;
    match pass {
        Pass::Inference => candle_nn::rotary_emb::rope(input, &cos, &sin),
        Pass::Differentiable => rope_composed(input, &cos, &sin),
    }
}

/// The rotary embedding written out of ops that have a backward pass.
///
/// Candle ships no differentiable rope: `candle_nn::rotary_emb::rope` ends in
/// `apply_op3_no_bwd` (candle-nn-0.11.0/src/rotary_emb.rs:580). Rather than
/// assume a convention, this reproduces the `RotaryEmb` CPU kernel in that same
/// file, which I read at lines 348-388. Passing a two-dimensional `cos`/`sin`
/// leaves the kernel's `unbatched_rope` flag false (line 349), and for every
/// batch and head it then walks `i_d` over `0..d/2` with
/// `i1 = i_t * d + i_d` (line 375), `i2 = i1 + d / 2` (line 376) and
/// `i_cs = i_t * (d / 2) + i_d` (line 377), writing:
///
/// ```text
/// dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];   // line 384
/// dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];   // line 385
/// ```
///
/// So this is the "rotate half" form: the last dimension splits into halves
/// `d / 2` apart, not adjacent interleaved pairs. `cos` and `sin` are `d / 2`
/// wide, indexed by position alone, and broadcast across batch and head. The
/// `t == 1` fast path (lines 352-367) is the same arithmetic with `i_t` pinned
/// to zero, so one expression covers both. Each output element is still exactly
/// one multiply, one multiply and one add or subtract, in that order, so in F32
/// the composed result is bit-comparable with the kernel rather than merely
/// close.
fn rope_composed(input: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, _, head_dim) = input.dims4()?;
    let half = head_dim / 2;
    let first = input.narrow(candle_core::D::Minus1, 0, half)?;
    let second = input.narrow(candle_core::D::Minus1, half, half)?;
    // `cos` and `sin` arrive as [sequence, d / 2]; two leading unit axes make
    // them broadcast over batch and head, which is what the kernel's `i_cs`
    // ignoring `bh_i` does by hand.
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let rotated_first = (first.broadcast_mul(&cos)? - second.broadcast_mul(&sin)?)?;
    let rotated_second = (first.broadcast_mul(&sin)? + second.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&rotated_first, &rotated_second], candle_core::D::Minus1)
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

/// The causal constraint and the key-padding constraint in one tensor.
///
/// A position is masked when it is in the query's future, `key > query`, or
/// when it is filler, `key >= lengths[row]`. The result is `[batch, 1,
/// sequence, sequence]`: one plane per row, and a unit head axis that
/// broadcasts, since every head of a row sees the same tokens.
///
/// Under the right padding [`SteeringLlama::forward_batch`] requires, the
/// second clause is redundant for a *real* query — `key <= query < length`
/// already implies `key < length` — and this says so rather than pretending
/// otherwise. What it buys is that the invariant holds by construction
/// instead of by coincidence of the padding side, and that a filler query,
/// whose row is computed whether or not anyone reads it, still mixes only
/// real keys. That second part is also why filler rows cannot produce a NaN:
/// row `query >= length` keeps keys `0..length`, which is never empty because
/// a zero length is refused.
///
/// Masked entries are `1`, matching [`Cache::mask`], so both feed the same
/// `masked_fill` and the same `f32::NEG_INFINITY`, which softmax turns into
/// exactly zero weight.
fn padded_causal_mask(
    lengths: &[usize],
    sequence: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut values = vec![0u8; lengths.len() * sequence * sequence];
    for (row, &length) in lengths.iter().enumerate() {
        let plane = row * sequence * sequence;
        for query in 0..sequence {
            let offset = plane + query * sequence;
            let visible = (query + 1).min(length);
            for slot in values[offset + visible..offset + sequence].iter_mut() {
                *slot = 1;
            }
        }
    }
    Tensor::from_vec(values, (lengths.len(), 1, sequence, sequence), device)
}

#[derive(Debug, Clone)]
struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
    gate_adapter: Option<Adapter>,
    up_adapter: Option<Adapter>,
    down_adapter: Option<Adapter>,
}

impl FeedForward {
    fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            gate: linear_no_bias(config.hidden_size, config.intermediate_size, builder.pp("gate_proj"))?,
            up: linear_no_bias(config.hidden_size, config.intermediate_size, builder.pp("up_proj"))?,
            down: linear_no_bias(config.intermediate_size, config.hidden_size, builder.pp("down_proj"))?,
            gate_adapter: adapters.get(layer, Target::Gate).cloned(),
            up_adapter: adapters.get(layer, Target::Up).cloned(),
            down_adapter: adapters.get(layer, Target::Down).cloned(),
        })
    }

    /// No `Pass` here — `silu` and the elementwise product both backpropagate,
    /// so the feed-forward block is already differentiable as written — but a
    /// `Route`, because its three projections are adapter sites like any other.
    fn forward(&self, hidden: &Tensor, route: Route) -> candle_core::Result<Tensor> {
        let gated =
            (candle_nn::ops::silu(&project(&self.gate, self.gate_adapter.as_ref(), hidden, route)?)?
                * project(&self.up, self.up_adapter.as_ref(), hidden, route)?)?;
        project(&self.down, self.down_adapter.as_ref(), &gated, route)
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
    fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            attention_norm: rms_norm(config.hidden_size, config.rms_norm_eps, builder.pp("input_layernorm"))?,
            attention: Attention::load(builder.pp("self_attn"), config, layer, adapters)?,
            feed_forward_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                builder.pp("post_attention_layernorm"),
            )?,
            feed_forward: FeedForward::load(builder.pp("mlp"), config, layer, adapters)?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let attention = self.attention.forward(
            &normalize(&self.attention_norm, hidden, mode.pass)?,
            index_pos,
            layer,
            cache,
            mask,
            mode,
        )?;
        let hidden = (hidden + attention)?;
        let feed_forward = self.feed_forward.forward(
            &normalize(&self.feed_forward_norm, &hidden, mode.pass)?,
            mode.route,
        )?;
        hidden + feed_forward
    }
}

/// RMS normalisation: fused for inference, composed for training.
///
/// `RmsNorm::forward` dispatches to `candle_nn::ops::rms_norm`, which ends in
/// `apply_op2_no_bwd` (candle-nn-0.11.0/src/ops.rs:684). `forward_diff`
/// (candle-nn-0.11.0/src/layer_norm.rs:197) is the same normalisation built
/// from `sqr`, `sum_keepdim`, `broadcast_div` and `broadcast_mul`, which do
/// record backward nodes.
fn normalize(norm: &RmsNorm, hidden: &Tensor, pass: Pass) -> candle_core::Result<Tensor> {
    match pass {
        Pass::Inference => norm.forward(hidden),
        Pass::Differentiable => norm.forward_diff(hidden),
    }
}

#[derive(Debug, Clone)]
pub struct SteeringLlama {
    embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    config: Config,
    adapters: Adapters,
}

impl SteeringLlama {
    pub fn load(builder: VarBuilder<'_>, config: Config) -> Result<Self> {
        Ok(Self::load_with_adapters(builder, config, Adapters::default())?)
    }

    /// Loads the frozen base and attaches `adapters`.
    ///
    /// The base weights come from `builder`, which Ster maps read-only out of
    /// safetensors; only the adapter factors were ever registered in a `VarMap`.
    /// Attaching them here therefore cannot make a base weight trainable, which
    /// is the property the whole training path rests on.
    pub fn load_with_adapters(
        builder: VarBuilder<'_>,
        config: Config,
        adapters: crate::lora::Adapters,
    ) -> candle_core::Result<Self> {
        let embeddings = embedding(config.vocab_size, config.hidden_size, builder.pp("model.embed_tokens"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, builder.pp("lm_head"))?
        };
        let final_norm = rms_norm(config.hidden_size, config.rms_norm_eps, builder.pp("model.norm"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                DecoderLayer::load(
                    builder.pp(format!("model.layers.{index}")),
                    &config,
                    index,
                    &adapters,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self { embeddings, layers, final_norm, lm_head, config, adapters })
    }

    pub fn adapters(&self) -> &crate::lora::Adapters {
        &self.adapters
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
        Ok(self.forward_pass(tokens, index_pos, cache, steering, capture_layers, Mode::DECODE)?)
    }

    /// `mode` picks the kernels, the adapter route and the readout; every
    /// other argument is unchanged.
    pub fn forward_pass(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
        mode: Mode,
    ) -> candle_core::Result<ForwardOutput> {
        self.decode(tokens, index_pos, cache, steering, capture_layers, None, mode)
    }

    /// A batch of unequal-length sequences in a single pass.
    ///
    /// `tokens` is `[batch, sequence]` and `lengths` says how many of each
    /// row's columns are real; everything past a row's length is filler the
    /// caller stacked to make the rectangle. The logits come back
    /// `[batch, sequence, vocab]` — a full rectangle, of which only the first
    /// `lengths[row]` rows of each slab mean anything.
    ///
    /// **Padding sits on the right**, and both consequences are load-bearing.
    /// The first is positional: with no key-value cache every column `j` is
    /// absolute position `j`, so right padding leaves every real token at the
    /// position it would have held alone, and the one shared rotary window
    /// `cos[0..sequence]` is correct for all rows at once. Left padding would
    /// shift each row's real tokens by its own pad count, which no scalar
    /// `index_pos` can express and which this decoder has no per-row position
    /// argument to carry. The second is what the caller must then do: a row's
    /// real logits are rows `0..lengths[row]` of its slab, so a loss slices
    /// each row from the front and stops at its own length rather than
    /// slicing a common window off the end. (Left padding is the right choice
    /// for batched *decoding*, where aligning every row's last real token at
    /// the final column is what lets one cached step serve the whole batch.
    /// This path has no cache and never decodes.)
    ///
    /// Refusals, rather than a plausible-looking loss over filler: a batch
    /// with no rows or no columns, a `lengths` that does not describe every
    /// row, a row claiming more tokens than the batch is wide, a row claiming
    /// none at all, a batch wider than the rotary tables, a readout of one
    /// last position, and a key-value cache.
    ///
    /// Activations are not captured here. Capture is defined as row zero's
    /// final column, which in a padded batch is filler, and a per-row capture
    /// is a different feature than the one the steering path asked for.
    pub fn forward_batch(
        &self,
        tokens: &Tensor,
        lengths: &[usize],
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        mode: Mode,
    ) -> Result<ForwardOutput> {
        let (batch, sequence) = tokens.dims2()?;
        if batch == 0 || sequence == 0 {
            bail!("a batched forward needs at least one row of at least one token");
        }
        if lengths.len() != batch {
            bail!("a batched forward got {} lengths for {batch} rows", lengths.len());
        }
        if cache.use_kv_cache {
            bail!("a batched forward cannot share one key-value cache across rows");
        }
        if sequence > self.config.max_position_embeddings {
            bail!(
                "a batch {sequence} tokens wide exceeds the {} positions this model was built for",
                self.config.max_position_embeddings
            );
        }
        if matches!(mode.readout, Readout::LastPosition) {
            bail!("a batched forward cannot read one last position, because every row ends somewhere else");
        }
        for (row, &length) in lengths.iter().enumerate() {
            if length == 0 {
                bail!("row {row} of the batch holds no tokens");
            }
            if length > sequence {
                bail!("row {row} claims {length} tokens in a batch only {sequence} wide");
            }
        }
        let mask = padded_causal_mask(lengths, sequence, tokens.device())?;
        Ok(self.decode(tokens, 0, cache, steering, &[], Some(&mask), mode)?)
    }

    /// The decoder loop both entry points run.
    ///
    /// `mask`, when present, replaces the cache's causal mask for every layer:
    /// a batched caller builds one combined causal and key-padding mask up
    /// front and hands the same handle down the stack, because the constraint
    /// depends only on the batch's lengths and so is identical at every one of
    /// the model's layers. `None` is the historical single-sequence path,
    /// which still asks the cache for a mask keyed by shape alone — a padding
    /// mask has no such key, since two batches of the same shape can pad
    /// differently, which is why it is built per call and never memoised.
    fn decode(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<ForwardOutput> {
        let (_, sequence) = tokens.dims2()?;
        let mut hidden = self.embeddings.forward(tokens)?;
        let mut activations = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, index_pos, index, cache, mask, mode)?;
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
                    hidden = hidden.broadcast_add(&scaled).map_err(|error| {
                        error.context(format!("failed to apply steering at layer {index}"))
                    })?;
                }
            }
        }
        let hidden = normalize(&self.final_norm, &hidden, mode.pass)?;
        // Decoding only ever samples the next token, so it projects one row and
        // leaves the rest of the vocabulary matmul undone. Anything that scores
        // a sequence against its own successors needs every position, and a
        // reward head needs no vocabulary at all.
        let logits = match mode.readout {
            Readout::LastPosition => {
                let last = hidden.i((.., sequence - 1, ..))?.contiguous()?;
                Some(self.lm_head.forward(&last)?.to_dtype(DType::F32)?)
            }
            Readout::EveryPosition => {
                Some(self.lm_head.forward(&hidden.contiguous()?)?.to_dtype(DType::F32)?)
            }
            Readout::Hidden => None,
        };
        Ok(ForwardOutput { logits, hidden: hidden.to_dtype(DType::F32)?, activations })
    }
}
