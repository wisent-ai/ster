//! model.rs — the decoder Ster reads, steers and trains through.
//!
//! # Where the casts are, and the one rule that puts them there
//!
//! The base weights are mapped at whatever dtype the loader chose — F32, F16
//! or BF16 — and the adapters, any head, and every optimizer moment are F32
//! regardless, because a low-rank correction smaller than the weight's own ulp
//! rounds to nothing in half. Inside the forward the rule is narrower than
//! "half everywhere" and can be stated in one line:
//!
//! **A tensor may be held at the weights' dtype. A sum over many terms is
//! taken in F32.**
//!
//! Half precision costs a mantissa, and a mantissa only matters where error
//! accumulates. A projection is one dot product against a weight that is
//! itself half — nothing is gained by widening it. A softmax, a norm, a rotary
//! angle and a log-probability all accumulate across a sequence, and each of
//! them is where a half mantissa turns into a wrong answer rather than a
//! slightly rounded one. So, site by site, with the reason each one is where
//! it is:
//!
//! * **Both masks are `u8` and therefore dtype-free.** [`Cache::mask`] and
//!   [`padded_causal_mask`] mark a key as hidden with a one, and
//!   [`masked_fill`] turns that into a negative infinity in the dtype the
//!   scores are already in. A mask has no precision to lose, and building one
//!   per dtype would be a way to get it wrong.
//! * **Attention scores and the softmax are F32, always.** Query, key and
//!   value are promoted before the score matmul and the result is cast back
//!   after the value matmul. The softmax sums an exponential over the whole
//!   key axis, which is the longest reduction in the pass and the one that
//!   grows with context; in F16 its accumulator saturates while the true value
//!   is still finite.
//! * **Rotary is F32 in and F32 through, and casts back.** The tables are held
//!   F32 whatever the weights are, because they are indexed by absolute
//!   position rather than derived from a weight: in F16 two neighbouring late
//!   positions round to the same angle, which rotates two different tokens
//!   identically. [`apply_rotary`] returns the rotated query and key at the
//!   weights' dtype, so the key-value cache still stores half-precision keys
//!   and half precision is still a memory saving. Measured, because the reason
//!   above is only a reason: TinyLlama-1.1B-Chat at F16, two held-out examples
//!   1864 and 1860 tokens long, scored by a build holding the tables and the
//!   rotation at F16 and by this one. The F32 rotation gives a corpus loss of
//!   1.7967694600423176 against the F32 model's own 1.795495907465617 — 0.07%
//!   — while rotating at F16 gives 1.81551726659139, or 1.1%, so this cast
//!   removes about fifteen sixteenths of what half precision otherwise costs
//!   at that length. Both builds agree to the last digit at `--precision f32`.
//!   At 128 tokens the two are indistinguishable, which is the honest limit of
//!   the claim: this matters as the context grows, and a short set cannot see
//!   it.
//! * **Norms promote themselves.** Both spellings — the fused
//!   `ops::rms_norm` and the composed `LayerNorm::forward` that training takes
//!   — cast F16 and BF16 up to F32 for the sum of squares and back afterwards
//!   (candle-nn-0.11.0/src/layer_norm.rs:123-138). This file adds nothing, and
//!   should not: a second promotion around a function that already promotes is
//!   a cast that reads as a safeguard and is really just a copy.
//! * **The readout leaves in F32.** Both the vocabulary projection and the
//!   residual stream are cast to F32 before they are returned, so every loss
//!   in `tune` is F32 arithmetic over an F32 input whatever the checkpoint was
//!   mapped at, and no objective has to know which precision it is training
//!   against.
//!
//! What stays at the weights' dtype is everything whose error does not
//! compound: the embedding lookup, the four attention projections, the three
//! feed-forward projections, the residual stream between layers, and the
//! key-value cache. That is where the bytes are, which is why mapping them
//! half is worth doing at all.
//!
//! Every cast above short-circuits to a handle clone when the dtypes already
//! agree (candle-core-0.11.0/src/tensor.rs:2453), so an F32 run records the
//! same ops it recorded before any of this existed — which is a claim the
//! product is expected to demonstrate by writing a byte-identical adapter, not
//! merely to assert here.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};

mod attention;
mod cache;
mod decoder;
mod layer;

pub use cache::Cache;
pub use decoder::SteeringLlama;

/// How a checkpoint's decoder differs from the plain Llama block.
///
/// Every family Ster loads is the same pre-norm, rotary, grouped-query
/// decoder with a gated feed-forward and Llama's tensor names; what changes
/// between them is a handful of switches, each read from the checkpoint's own
/// `config.json` rather than assumed:
///
/// * **Head width** — `head_dim` when the config states it (Qwen3, Gemma),
///   otherwise `hidden_size / num_attention_heads`. Attention may then be
///   wider or narrower than the residual stream.
/// * **Per-head query and key norms** — Qwen3's `self_attn.q_norm` and
///   `self_attn.k_norm`, applied after the projection and before the rotary
///   embedding.
/// * **Projection bias** — Qwen2 adds a bias to query, key and value; a Llama
///   or Mistral config with `attention_bias` adds one to the output too.
/// * **Sliding-window attention** — a query on a local layer sees only the
///   `sliding_window` keys behind it. Which layers are local comes from the
///   config's `layer_types` when it lists them, otherwise from the family:
///   every layer for Mistral, layers from `max_window_layers` on for Qwen2 and
///   Qwen3 with `use_sliding_window`, every even layer for Gemma 2.
/// * **Fused projections** — Phi-3 stores query, key and value as one
///   `qkv_proj` and the gate and up projections as one `gate_up_proj`; each
///   is split back into the separate projections at load.
/// * **Gemma's conventions** — every RMS norm scales by `1 + weight`, the
///   embedding is multiplied by `sqrt(hidden_size)`, the feed-forward gate is
///   the tanh-approximated GELU, and the word embeddings are tied.
/// * **Gemma 2's additions** — a norm after attention and another after the
///   feed-forward, scores scaled by `query_pre_attn_scalar` instead of the head
///   width, and `tanh` soft-capping of attention scores and final logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Architecture {
    pub head_dim: usize,
    pub query_key_norm: bool,
    pub query_key_value_bias: bool,
    pub output_bias: bool,
    pub sliding_window: Option<usize>,
    /// Bit `i` set means layer `i` attends through the sliding window.
    pub sliding_layers: u128,
    pub fused_projections: bool,
    pub norm_offset: bool,
    pub embedding_scale: bool,
    pub activation: Activation,
    pub sandwich_norms: bool,
    /// What attention scores are divided by: `sqrt(head_dim)`, or Gemma 2's
    /// `sqrt(query_pre_attn_scalar)`.
    pub score_divisor: f64,
    pub attention_softcap: Option<f64>,
    pub final_softcap: Option<f64>,
}

impl Architecture {
    /// The plain Llama block with a head width derived from the residual one.
    pub fn llama(hidden_size: usize, heads: usize) -> Self {
        let head_dim = hidden_size / heads;
        Self {
            head_dim,
            query_key_norm: false,
            query_key_value_bias: false,
            output_bias: false,
            sliding_window: None,
            sliding_layers: 0,
            fused_projections: false,
            norm_offset: false,
            embedding_scale: false,
            activation: Activation::Silu,
            sandwich_norms: false,
            score_divisor: (head_dim as f64).sqrt(),
            attention_softcap: None,
            final_softcap: None,
        }
    }

    /// The window layer `layer` attends through, or `None` for full attention.
    pub fn window(&self, layer: usize) -> Option<usize> {
        let sliding = layer < 128 && self.sliding_layers & (1u128 << layer) != 0;
        self.sliding_window.filter(|_| sliding)
    }

    /// Width of the query projection and of the attention output.
    pub fn attention_width(&self, heads: usize) -> usize {
        heads * self.head_dim
    }
}

/// The non-linearity on the feed-forward gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    /// `gelu_pytorch_tanh`, which Candle's `gelu` computes.
    GeluTanh,
}

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
        Self {
            pass: Pass::Inference,
            route,
            readout: Readout::EveryPosition,
        }
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
        Ok(Self {
            vectors: tensors,
            strength,
            hidden_size,
        })
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
