//! Grouped-query attention with adapters on its projections: the scores, the
//! rotary angles they are taken at, and the masks that hide what a position
//! may not see.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear, linear_no_bias};
use candle_transformers::models::llama::Config;

use crate::lora::{Adapter, Adapters, Target};

use super::{
    Architecture, Cache, Mode, Pass, QueryKeyNorm, Route,
    layer::{load_norm, normalize},
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
    /// (Qwen3, Gemma 3) or over the whole projection (OLMo 2). `None` on a
    /// family that has neither.
    query_norm: Option<RmsNorm>,
    key_norm: Option<RmsNorm>,
    query_key_norm: QueryKeyNorm,
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    /// How many keys behind it a query may see, on a sliding-window layer.
    window: Option<usize>,
    rotary: Rotary,
    score_divisor: f64,
    softcap: Option<f64>,
}

/// Which rotary table this layer rotates its query and key with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rotary {
    Global,
    /// The sliding-window table (Gemma 3's `rope_local_base_freq`).
    Local,
    /// No rotation at all (SmolLM3's NoPE layers).
    None,
}

/// A projection, with the bias the architecture says this one carries.
fn projection(
    inputs: usize,
    outputs: usize,
    bias: bool,
    builder: VarBuilder<'_>,
) -> candle_core::Result<Linear> {
    if bias {
        linear(inputs, outputs, builder)
    } else {
        linear_no_bias(inputs, outputs, builder)
    }
}

impl Attention {
    pub(super) fn load(
        builder: VarBuilder<'_>,
        config: &Config,
        architecture: &Architecture,
        layer: usize,
        adapters: &Adapters,
    ) -> candle_core::Result<Self> {
        let input = config.hidden_size;
        let head_dim = architecture.head_dim;
        let query_width = architecture.attention_width(config.num_attention_heads);
        let key_value_width = head_dim * config.num_key_value_heads;
        let bias = architecture.query_key_value_bias;
        let norm = |width: usize, name: &str| {
            load_norm(width, config.rms_norm_eps, architecture.norm_offset, builder.pp(name))
        };
        let (query_norm, key_norm) = match architecture.query_key_norm {
            QueryKeyNorm::None => (None, None),
            QueryKeyNorm::PerHead => (Some(norm(head_dim, "q_norm")?), Some(norm(head_dim, "k_norm")?)),
            QueryKeyNorm::Full => (
                Some(norm(query_width, "q_norm")?),
                Some(norm(key_value_width, "k_norm")?),
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
        // Phi-3 stores query, key and value as one `qkv_proj` matrix, rows in
        // that order. Each projection is a row slice of it — a view of the
        // mapped weight, not a copy — so every adapter site stays separate.
        let (query, key, value) = if architecture.fused_projections {
            let fused = builder.get(
                (query_width + 2 * key_value_width, input),
                "qkv_proj.weight",
            )?;
            (
                Linear::new(fused.narrow(0, 0, query_width)?, None),
                Linear::new(fused.narrow(0, query_width, key_value_width)?, None),
                Linear::new(
                    fused.narrow(0, query_width + key_value_width, key_value_width)?,
                    None,
                ),
            )
        } else {
            (
                projection(input, query_width, bias, builder.pp("q_proj"))?,
                projection(input, key_value_width, bias, builder.pp("k_proj"))?,
                projection(input, key_value_width, bias, builder.pp("v_proj"))?,
            )
        };
        Ok(Self {
            query,
            key,
            value,
            output: projection(
                query_width,
                input,
                architecture.output_bias,
                builder.pp("o_proj"),
            )?,
            query_adapter: adapters.get(layer, Target::Query).cloned(),
            key_adapter: adapters.get(layer, Target::Key).cloned(),
            value_adapter: adapters.get(layer, Target::Value).cloned(),
            output_adapter: adapters.get(layer, Target::Output).cloned(),
            query_norm,
            key_norm,
            query_key_norm: architecture.query_key_norm,
            heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim,
            window,
            rotary,
            score_divisor: architecture.score_divisor,
            softcap: architecture.attention_softcap,
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
        // heads; Qwen3 and Gemma 3 normalise each head after the split.
        let (query, key) = if self.query_key_norm == QueryKeyNorm::Full {
            (
                per_head_norm(self.query_norm.as_ref(), query, mode.pass)?,
                per_head_norm(self.key_norm.as_ref(), key, mode.pass)?,
            )
        } else {
            (query, key)
        };
        let query = query.reshape((batch, sequence, self.heads, self.head_dim))?;
        let key = key.reshape((batch, sequence, self.key_value_heads, self.head_dim))?;
        let (query, key) = if self.query_key_norm == QueryKeyNorm::PerHead {
            (
                per_head_norm(self.query_norm.as_ref(), query, mode.pass)?,
                per_head_norm(self.key_norm.as_ref(), key, mode.pass)?,
            )
        } else {
            (query, key)
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
        let query = match tables {
            Some((cos, sin)) => apply_rotary(&query, index_pos, cos, sin, cache.weights, mode.pass)?,
            None => query,
        };
        if let Some((cos, sin)) = tables {
            key = apply_rotary(&key, index_pos, cos, sin, cache.weights, mode.pass)?;
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
/// `head_dim` after the split (Qwen3, Gemma 3), or the whole projection before
/// it (OLMo 2). Either way it happens before the rotary embedding. A family
/// without the norm passes the projection through untouched.
fn per_head_norm(norm: Option<&RmsNorm>, heads: Tensor, pass: Pass) -> candle_core::Result<Tensor> {
    match norm {
        Some(norm) => normalize(norm, &heads, pass),
        None => Ok(heads),
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
/// StableLM do.
fn apply_rotary(
    input: &Tensor,
    index_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    weights: DType,
    pass: Pass,
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
    let rotated = match pass {
        Pass::Inference => candle_nn::rotary_emb::rope(&rotating.contiguous()?, &cos, &sin)?,
        Pass::Differentiable => rope_composed(&rotating, &cos, &sin)?,
    };
    let rotated = match kept {
        Some(kept) => Tensor::cat(&[&rotated, &kept], 3)?,
        None => rotated,
    };
    rotated.to_dtype(weights)
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
