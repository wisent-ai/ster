//! Grouped-query attention with adapters on its projections: the scores, the
//! rotary angles they are taken at, and the masks that hide what a position
//! may not see.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::{
    Architecture, Cache, Mode, NormKind, Pass, Positions, QkvLayout, QueryKeyNorm, Route,
    layer::{
        norm::{Norm, NormSpec},
        projection,
    },
};

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
    /// Query and key norms, applied before the rotary embedding: per head
    /// (Qwen3, Gemma 3, Cohere, StableLM) or over the whole projection
    /// (OLMo 2). `None` on a family that has neither.
    query_norm: Option<Norm>,
    key_norm: Option<Norm>,
    query_key_norm: QueryKeyNorm,
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    /// How many keys behind it a query may see, on a sliding-window layer.
    window: Option<usize>,
    rotary: Rotary,
    /// Rotate adjacent pairs (Cohere) instead of halves.
    interleaved: bool,
    score_divisor: f64,
    softcap: Option<f64>,
    /// ALiBi's slope per head, `[1, heads, 1, 1]`, for a family whose
    /// positions are a linear bias on the scores (BLOOM, MPT).
    alibi: Option<Tensor>,
}

/// Which rotary table this layer rotates its query and key with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rotary {
    Global,
    /// The sliding-window table (Gemma 3's `rope_local_base_freq`).
    Local,
    /// No rotation at all (SmolLM3's NoPE layers, Cohere 2's global layers).
    None,
}

impl Attention {
    /// `builder` is the layer's; the projections sit below the family's
    /// attention block (`self_attn`, `attention`, `attn`).
    pub(super) fn load(
        layer_builder: &VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let names = architecture.names;
        let builder = layer_builder.pp(names.attention);
        let input = config.hidden_size;
        let heads = config.num_attention_heads;
        let key_value_heads = config.num_key_value_heads;
        let head_dim = architecture.head_dim;
        let query_width = architecture.attention_width(heads);
        let key_value_width = head_dim * key_value_heads;
        let bias = architecture.query_key_value_bias;
        let conv1d = architecture.conv1d;
        // A LayerNorm family's query and key norms carry no bias; the others
        // are RMS norms like the rest of the block.
        let spec = NormSpec {
            kind: match architecture.norm {
                NormKind::Layer { .. } => NormKind::Layer { bias: false },
                other => other,
            },
            ..NormSpec::of(architecture)
        };
        let (query_norm, key_norm) = match architecture.query_key_norm {
            QueryKeyNorm::None => (None, None),
            QueryKeyNorm::PerHead => (
                Some(spec.load(head_dim, builder.pp("q_norm"))?),
                Some(spec.load(head_dim, builder.pp("k_norm"))?),
            ),
            QueryKeyNorm::Full => (
                Some(spec.load(query_width, builder.pp("q_norm"))?),
                Some(spec.load(key_value_width, builder.pp("k_norm"))?),
            ),
            QueryKeyNorm::HeadWeights => (
                Some(spec.load_head_weights(heads, head_dim, builder.pp("q_norm"))?),
                Some(spec.load_head_weights(key_value_heads, head_dim, builder.pp("k_norm"))?),
            ),
            QueryKeyNorm::HeadModules => (
                Some(spec.load_head_modules(heads, head_dim, builder.pp("q_layernorm"))?),
                Some(spec.load_head_modules(key_value_heads, head_dim, builder.pp("k_layernorm"))?),
            ),
        };
        let window = architecture.window(layer);
        let rotary = if !architecture.rotates(layer) {
            Rotary::None
        } else if window.is_some() && architecture.local_rope_theta.is_some() {
            Rotary::Local
        } else {
            Rotary::Global
        };
        let (query, key, value) = match architecture.qkv_layout {
            QkvLayout::Separate => (
                projection(input, query_width, bias, conv1d, builder.pp(names.query))?,
                projection(input, key_value_width, bias, conv1d, builder.pp(names.key))?,
                projection(input, key_value_width, bias, conv1d, builder.pp(names.value))?,
            ),
            // Phi-3, GPT-2 and GPT-BigCode store query, key and value as one
            // matrix, rows in that order. Each projection is a row slice of it
            // — a view of the mapped weight, not a copy — so every adapter
            // site stays separate. GPT-2's `Conv1D` stores it transposed, so
            // that one is laid out once at load.
            QkvLayout::Stacked => {
                let rows = query_width + 2 * key_value_width;
                let fused = builder.pp(names.fused_qkv);
                let weight = if conv1d {
                    fused.get((input, rows), "weight")?.t()?.contiguous()?
                } else {
                    fused.get((rows, input), "weight")?
                };
                let bias = if bias { Some(fused.get(rows, "bias")?) } else { None };
                let slice = |start: usize, width: usize| -> candle_core::Result<Linear> {
                    Ok(Linear::new(
                        weight.narrow(0, start, width)?,
                        bias.as_ref().map(|bias| bias.narrow(0, start, width)).transpose()?,
                    ))
                };
                (
                    slice(0, query_width)?,
                    slice(query_width, key_value_width)?,
                    slice(query_width + key_value_width, key_value_width)?,
                )
            }
            // GPT-NeoX, BLOOM and Falcon lay the rows out by key-value group:
            // each group's query heads, then its key head, then its value
            // head. With as many groups as heads (GPT-NeoX, BLOOM) that is
            // each head's query, key and value in turn. Gathering a
            // projection's rows is a copy, made once at load.
            QkvLayout::Grouped => {
                let per_group = heads / key_value_heads;
                let rows = query_width + 2 * key_value_width;
                let fused = builder.pp(names.fused_qkv);
                let weight = fused
                    .get((rows, input), "weight")?
                    .reshape((key_value_heads, per_group + 2, head_dim, input))?;
                let bias = if bias {
                    Some(fused.get(rows, "bias")?.reshape((key_value_heads, per_group + 2, head_dim))?)
                } else {
                    None
                };
                let part = |start: usize, count: usize| -> candle_core::Result<Linear> {
                    let width = key_value_heads * count * head_dim;
                    Ok(Linear::new(
                        weight.narrow(1, start, count)?.contiguous()?.reshape((width, input))?,
                        bias.as_ref()
                            .map(|bias| bias.narrow(1, start, count)?.contiguous()?.reshape(width))
                            .transpose()?,
                    ))
                };
                (part(0, per_group)?, part(per_group, 1)?, part(per_group + 1, 1)?)
            }
        };
        Ok(Self {
            query,
            key,
            value,
            output: projection(
                query_width,
                input,
                architecture.output_bias,
                conv1d,
                layer_builder.pp(architecture.names.output),
            )?,
            query_adapter: adapters.get(layer, Target::Query).cloned(),
            key_adapter: adapters.get(layer, Target::Key).cloned(),
            value_adapter: adapters.get(layer, Target::Value).cloned(),
            output_adapter: adapters.get(layer, Target::Output).cloned(),
            query_norm,
            key_norm,
            query_key_norm: architecture.query_key_norm,
            heads,
            key_value_heads,
            head_dim,
            window,
            rotary,
            interleaved: architecture.interleaved_rotary,
            score_divisor: architecture.score_divisor,
            softcap: architecture.attention_softcap,
            alibi: match architecture.positions {
                // Falcon adds the bias before dividing the scores, so its
                // slopes are divided too.
                Positions::Alibi { inside_scale } => {
                    let scale = if inside_scale { 1.0 / architecture.score_divisor } else { 1.0 };
                    Some(
                        (Tensor::new(alibi_slopes(heads), layer_builder.device())?
                            .reshape((1, heads, 1, 1))?
                            * scale)?,
                    )
                }
                _ => None,
            },
        })
    }

    /// The window this layer attends through, if it is a sliding-window layer.
    pub(super) fn window(&self) -> Option<usize> {
        self.window
    }

    /// One attention block, optionally under a caller-supplied mask.
    ///
    /// `mask` is the combined causal and key-padding constraint a batched
    /// caller built once for the whole stack, shaped `[batch, 1, sequence,
    /// keys]` so the unit head axis broadcasts across every head. When it is
    /// `None` this is the single-sequence path and the mask comes from the
    /// cache, exactly as it always did.
    pub(super) fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        let query = project(&self.query, self.query_adapter.as_ref(), hidden, mode.route)?;
        let key = project(&self.key, self.key_adapter.as_ref(), hidden, mode.route)?;
        // OLMo 2 normalises the whole projection before it is split into
        // heads; every other query and key norm works per head after it.
        let (query, key) = if self.query_key_norm == QueryKeyNorm::Full {
            (
                optional_norm(self.query_norm.as_ref(), query, mode.pass)?,
                optional_norm(self.key_norm.as_ref(), key, mode.pass)?,
            )
        } else {
            (query, key)
        };
        let query = query.reshape((batch, sequence, self.heads, self.head_dim))?;
        let key = key.reshape((batch, sequence, self.key_value_heads, self.head_dim))?;
        let (query, key) = if self.query_key_norm == QueryKeyNorm::Full {
            (query, key)
        } else {
            (
                optional_norm(self.query_norm.as_ref(), query, mode.pass)?,
                optional_norm(self.key_norm.as_ref(), key, mode.pass)?,
            )
        };
        let query = query.transpose(1, 2)?.contiguous()?;
        let mut key = key.transpose(1, 2)?.contiguous()?;
        let mut value = project(&self.value, self.value_adapter.as_ref(), hidden, mode.route)?
            .reshape((batch, sequence, self.key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let tables = match self.rotary {
            // LongRoPE switches every position to the long factors once the
            // sequence runs past the original context, as Phi-3 does.
            Rotary::Global => match &cache.long {
                Some((cos, sin, original)) if index_pos + sequence > *original => Some((cos, sin)),
                _ => Some((&cache.cos, &cache.sin)),
            },
            Rotary::Local => cache.local.as_ref().map(|(cos, sin)| (cos, sin)),
            Rotary::None => None,
        };
        let rotate = |input: &Tensor, (cos, sin): (&Tensor, &Tensor)| {
            apply_rotary(input, index_pos, cos, sin, cache.weights, mode.pass, self.interleaved)
        };
        let query = match tables {
            Some(tables) => rotate(&query, tables)?,
            None => query,
        };
        if let Some(tables) = tables {
            key = rotate(&key, tables)?;
        }
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
        let attention = (query.matmul(&key.t()?)? / self.score_divisor)?;
        // ALiBi adds `slope * key_position` to every score; within one query's
        // row that is the published `-slope * distance` plus a constant the
        // softmax ignores, and it is what BLOOM computes.
        let attention = match &self.alibi {
            Some(slopes) => {
                let keys = attention.dim(candle_core::D::Minus1)?;
                let positions = Tensor::arange(0u32, keys as u32, attention.device())?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, 1, keys))?;
                attention.broadcast_add(&slopes.broadcast_mul(&positions)?)?
            }
            None => attention,
        };
        // Gemma 2 bounds every score to `(-cap, cap)` with a tanh before the
        // mask, so no single key can take the whole softmax.
        let attention = match self.softcap {
            Some(cap) => ((attention / cap)?.tanh()? * cap)?,
            None => attention,
        };
        let keys = attention.dim(candle_core::D::Minus1)?;
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
            // already exist, so there is nothing causality would remove — and
            // nothing a window would either, while every key is inside it.
            None if sequence == 1 && self.window.is_none_or(|window| keys <= window) => attention,
            None => {
                let mask = cache
                    .mask(sequence, index_pos, self.window)?
                    .broadcast_as(attention.shape())?;
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
        let output = attention
            .matmul(&value.contiguous()?)?
            .to_dtype(input_dtype)?;
        let output = output
            .transpose(1, 2)?
            .reshape((batch, sequence, self.heads * self.head_dim))?;
        project(
            &self.output,
            self.output_adapter.as_ref(),
            &output,
            mode.route,
        )
    }
}

/// Normalizes the last axis when the architecture has the norm: one head's
/// `head_dim` after the split, or the whole projection before it (OLMo 2).
/// Either way it happens before the rotary embedding. A family without the
/// norm passes the projection through untouched.
fn optional_norm(norm: Option<&Norm>, input: Tensor, pass: Pass) -> candle_core::Result<Tensor> {
    match norm {
        Some(norm) => norm.forward(&input, pass),
        None => Ok(input),
    }
}

/// Applies a projection, adding the low-rank update when this site is adapted
/// and `route` asks for it.
///
/// The `None` arm is the historical code path down to the op: one nullable
/// check, no tensor allocated, no dtype touched. An unadapted model — the whole
/// steering product — therefore costs a null-pointer test per projection, and
/// a reference pass over an adapted model costs one enum comparison more.
pub(super) fn project(
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

/// Rotates `input` by the angles at `index_pos..index_pos + sequence`, in F32,
/// and returns the result at `weights`.
///
/// The rotation is a multiply-add against a table indexed by absolute
/// position, so it is the one place in the forward where a half-precision
/// mantissa is spent on something other than a weight: at F16 the angles
/// themselves collide late in the context, and the phase error it introduces
/// looks like a slightly different sentence rather than like a numerical
/// fault. F32 in, F32 through, `weights` out — the cast back is what keeps a
/// half-precision key-value cache half-precision, since the key this returns
/// is the key the cache stores.
///
/// At F32 every cast here short-circuits to a handle clone
/// (candle-core-0.11.0/src/tensor.rs:2453), so the F32 path is unchanged down
/// to the op it records.
///
/// The tables are `rotary_dim / 2` wide. When that is narrower than the head
/// (a config's `partial_rotary_factor`), only the head's first `rotary_dim`
/// components rotate and the rest pass through, as Phi-4-mini, GPT-NeoX and
/// StableLM do. `interleaved` rotates adjacent pairs `(2i, 2i + 1)` instead
/// of halves `d / 2` apart, as Cohere does.
fn apply_rotary(
    input: &Tensor,
    index_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    weights: DType,
    pass: Pass,
    interleaved: bool,
) -> candle_core::Result<Tensor> {
    let (_, _, sequence, head_dim) = input.dims4()?;
    let rotary_dim = 2 * cos.dim(1)?;
    let cos = cos.narrow(0, index_pos, sequence)?;
    let sin = sin.narrow(0, index_pos, sequence)?;
    let input = input.to_dtype(DType::F32)?;
    let (rotating, kept) = if rotary_dim < head_dim {
        (
            input.narrow(3, 0, rotary_dim)?,
            Some(input.narrow(3, rotary_dim, head_dim - rotary_dim)?),
        )
    } else {
        (input, None)
    };
    let rotated = match (pass, interleaved) {
        (Pass::Inference, false) => {
            candle_nn::rotary_emb::rope(&rotating.contiguous()?, &cos, &sin)?
        }
        (Pass::Inference, true) => {
            candle_nn::rotary_emb::rope_i(&rotating.contiguous()?, &cos, &sin)?
        }
        (Pass::Differentiable, false) => rope_composed(&rotating, &cos, &sin)?,
        (Pass::Differentiable, true) => rope_interleaved_composed(&rotating, &cos, &sin)?,
    };
    let rotated = match kept {
        Some(kept) => Tensor::cat(&[&rotated, &kept], 3)?,
        None => rotated,
    };
    rotated.to_dtype(weights)
}

/// The interleaved rotation written out of ops that have a backward pass:
/// `candle_nn::rotary_emb::rope_i` computes, for each pair `(2i, 2i + 1)`,
/// `x[2i] * cos[i] - x[2i+1] * sin[i]` and `x[2i] * sin[i] + x[2i+1] * cos[i]`,
/// and ends in a no-backward op like `rope`. The head is viewed as `d / 2`
/// pairs, the two members rotated, and the pairs laid back out.
fn rope_interleaved_composed(
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_core::Result<Tensor> {
    let (batch, heads, sequence, head_dim) = input.dims4()?;
    let pairs = input.reshape((batch, heads, sequence, head_dim / 2, 2))?;
    let even = pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let rotated_even = (even.broadcast_mul(&cos)? - odd.broadcast_mul(&sin)?)?;
    let rotated_odd = (even.broadcast_mul(&sin)? + odd.broadcast_mul(&cos)?)?;
    Tensor::stack(&[&rotated_even, &rotated_odd], 4)?.reshape((batch, heads, sequence, head_dim))
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

/// The exponent range of ALiBi's geometric slopes: for `n` heads (a power of
/// two) the slopes are `2^(-span/n)` raised to `1..=n` (Press et al., 2022,
/// and Transformers' `build_alibi_tensor`).
pub(crate) const ALIBI_SPAN: f64 = 8.0;

/// ALiBi's slope for each of `heads` heads. A head count that is not a power
/// of two takes the slopes of the nearest power below, then every other slope
/// of twice that many, as BLOOM does.
fn alibi_slopes(heads: usize) -> Vec<f32> {
    let closest = 1usize << heads.ilog2();
    let base = (-ALIBI_SPAN / closest as f64).exp2();
    let mut slopes: Vec<f32> = (1..=closest).map(|power| base.powi(power as i32) as f32).collect();
    if closest != heads {
        let extra = (-ALIBI_SPAN / (2 * closest) as f64).exp2();
        let remaining = closest.min(heads - closest);
        slopes.extend((0..remaining).map(|index| extra.powi((2 * index + 1) as i32) as f32));
    }
    slopes
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
/// On a sliding-window layer a real query also loses every key `window` or
/// more positions behind it. A filler query keeps all real keys instead: a
/// window past the end of its row could leave it none, and a row of negative
/// infinities is a NaN that a training step would carry back into the
/// adapters even though no loss reads that row.
///
/// Masked entries are `1`, matching [`Cache::mask`], so both feed the same
/// `masked_fill` and the same `f32::NEG_INFINITY`, which softmax turns into
/// exactly zero weight.
pub(super) fn padded_causal_mask(
    lengths: &[usize],
    sequence: usize,
    window: Option<usize>,
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
            if query < length {
                for key in 0..visible {
                    if super::cache::hidden_key(query, key, window) {
                        values[offset + key] = 1;
                    }
                }
            }
        }
    }
    Tensor::from_vec(values, (lengths.len(), 1, sequence, sequence), device)
}
