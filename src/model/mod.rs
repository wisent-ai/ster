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
use candle_transformers::models::llama::Config;

mod attention;
mod cache;
mod decoder;
mod layer;

pub use cache::Cache;
pub use decoder::SteeringLlama;

use crate::lora::Target;

/// How a checkpoint's decoder differs from the plain Llama block.
///
/// Every family Ster loads is a rotary, grouped-query decoder; what changes
/// between them is a handful of switches and tensor names, each read from the
/// checkpoint's own `config.json` rather than assumed:
///
/// * **Head width** — `head_dim` when the config states it (Qwen3, Gemma),
///   otherwise `hidden_size / num_attention_heads`. Attention may then be
///   wider or narrower than the residual stream.
/// * **Query and key norms** — Qwen3 and Gemma 3 normalise each head of the
///   query and key (`self_attn.q_norm`, `self_attn.k_norm` over `head_dim`);
///   OLMo 2 normalises the whole query and key projection at once. Both come
///   after the projection and before the rotary embedding.
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
/// * **Post-norm blocks** — OLMo 2 has no norm before attention or the
///   feed-forward; it normalises each sublayer's output before the residual
///   add instead.
/// * **Granite's scales** — `embedding_multiplier` on the embedding,
///   `residual_multiplier` on each sublayer's output, `attention_multiplier`
///   in place of `1 / sqrt(head_dim)`, and final logits divided by
///   `logits_scaling`.
/// * **Rotary per layer** — SmolLM3 skips the rotary embedding on the layers
///   its `no_rope_layers` marks; Gemma 3 rotates local layers with
///   `rope_local_base_freq` and global ones with `rope_theta`.
/// * **Partial rotation** — `partial_rotary_factor` rotates only the first
///   share of each head (Phi-4-mini); the rest passes through.
/// * **Rotary scaling** — `linear` divides every angle by a factor; Phi-3's
///   `longrope` rescales each frequency by a stated factor, a short list
///   inside the original context and a long one beyond it.
/// * **Norm kind** — RMS for most families; LayerNorm, with a bias
///   (StableLM, Starcoder2, Phi-2, Nemotron) or without (Cohere), for others.
/// * **Feed-forward kind** — a gated feed-forward for most; a plain
///   up-activation-down one for Starcoder2, Phi-2 and Nemotron, with its own
///   tensor names and optional bias.
/// * **Parallel blocks** — Cohere, Phi-2 and StableLM with
///   `use_parallel_residual` feed one normalised input to attention and the
///   feed-forward and add both outputs to the residual at once.
/// * **Interleaved rotation** — Cohere rotates adjacent pairs rather than
///   halves.
#[derive(Debug, Clone, PartialEq)]
pub struct Architecture {
    pub head_dim: usize,
    /// How many components of each head rotate: `head_dim` unless the config
    /// states a `partial_rotary_factor`.
    pub rotary_dim: usize,
    pub query_key_norm: QueryKeyNorm,
    pub query_key_value_bias: bool,
    pub output_bias: bool,
    /// Bias on the feed-forward projections (Starcoder2, Phi-2, Nemotron's
    /// `mlp_bias`).
    pub feed_forward_bias: bool,
    pub feed_forward: FeedForwardKind,
    pub norm: NormKind,
    /// The norm epsilon, read from whichever key the family spells it with.
    pub norm_eps: f64,
    pub parallel: bool,
    /// A parallel block whose feed-forward has its own norm (GPT-NeoX)
    /// rather than sharing attention's (Cohere, Phi-2, GPT-J).
    pub parallel_norms: bool,
    pub interleaved_rotary: bool,
    /// A bias on the vocabulary projection (Phi-2).
    pub lm_head_bias: bool,
    pub names: Names,
    pub sliding_window: Option<usize>,
    /// Bit `i` set means layer `i` attends through the sliding window.
    pub sliding_layers: u128,
    /// Bit `i` set means layer `i` applies no rotary embedding.
    pub unrotated_layers: u128,
    /// The rotary base sliding-window layers use, when it differs from
    /// `rope_theta` (Gemma 3).
    pub local_rope_theta: Option<f32>,
    /// A rotary scaling Ster applies itself; Llama 3's is carried by Candle's
    /// config instead.
    pub rope_scaling: RopeScaling,
    pub qkv_layout: QkvLayout,
    /// Gate and up stored as one `gate_up_proj`, gate rows first (Phi-3,
    /// GLM).
    pub fused_feed_forward: bool,
    pub norm_offset: bool,
    /// What the embedding is multiplied by before the first block.
    pub embedding_multiplier: Option<f64>,
    /// What each sublayer's output is multiplied by before the residual add.
    pub residual_multiplier: Option<f64>,
    pub activation: Activation,
    /// A norm before attention and before the feed-forward (every family but
    /// OLMo 2).
    pub pre_norms: bool,
    /// A norm over attention's and the feed-forward's outputs (Gemma 2 and 3,
    /// OLMo 2).
    pub output_norms: bool,
    /// What attention scores are divided by: `sqrt(head_dim)`, Gemma 2's
    /// `sqrt(query_pre_attn_scalar)`, or Granite's `1 / attention_multiplier`.
    pub score_divisor: f64,
    pub attention_softcap: Option<f64>,
    pub final_softcap: Option<f64>,
    /// What final logits are multiplied by: Cohere's `logit_scale`, or one
    /// over Granite's `logits_scaling`.
    pub logits_multiplier: Option<f64>,
    /// A routed feed-forward in place of the dense one, on the layers it
    /// covers.
    pub experts: Option<MixtureOfExperts>,
}

impl Architecture {
    /// The plain Llama block with a head width derived from the residual one.
    pub fn llama(hidden_size: usize, heads: usize, norm_eps: f64) -> Self {
        let head_dim = hidden_size / heads;
        Self {
            head_dim,
            rotary_dim: head_dim,
            query_key_norm: QueryKeyNorm::None,
            query_key_value_bias: false,
            output_bias: false,
            feed_forward_bias: false,
            feed_forward: FeedForwardKind::Gated,
            norm: NormKind::Rms,
            norm_eps,
            parallel: false,
            parallel_norms: false,
            interleaved_rotary: false,
            lm_head_bias: false,
            names: Names::LLAMA,
            sliding_window: None,
            sliding_layers: 0,
            unrotated_layers: 0,
            local_rope_theta: None,
            rope_scaling: RopeScaling::None,
            qkv_layout: QkvLayout::Separate,
            fused_feed_forward: false,
            norm_offset: false,
            embedding_multiplier: None,
            residual_multiplier: None,
            activation: Activation::Silu,
            pre_norms: true,
            output_norms: false,
            score_divisor: (head_dim as f64).sqrt(),
            attention_softcap: None,
            final_softcap: None,
            logits_multiplier: None,
            experts: None,
        }
    }

    /// The window layer `layer` attends through, or `None` for full attention.
    pub fn window(&self, layer: usize) -> Option<usize> {
        let sliding = layer < 128 && self.sliding_layers & (1u128 << layer) != 0;
        self.sliding_window.filter(|_| sliding)
    }

    /// Whether layer `layer` applies the rotary embedding.
    pub fn rotates(&self, layer: usize) -> bool {
        layer >= 128 || self.unrotated_layers & (1u128 << layer) == 0
    }

    /// Width of the query projection and of the attention output.
    pub fn attention_width(&self, heads: usize) -> usize {
        heads * self.head_dim
    }

    /// Whether layer `layer`'s feed-forward is the mixture of experts.
    pub fn routed(&self, layer: usize) -> bool {
        self.experts.as_ref().is_some_and(|experts| {
            layer >= 128 || experts.dense_layers & (1u128 << layer) == 0
        })
    }

    /// Where `target`'s weight lives at `layer`, or `None` when this family
    /// has no single such projection: the gate of a plain feed-forward, or
    /// any feed-forward projection of a mixture of experts.
    ///
    /// A fused tensor holds several projections; the placement then names the
    /// row blocks this projection owns, so merging adds the update to those
    /// rows alone. Phi-3 stacks query, key and value; GPT-NeoX interleaves
    /// them head by head; Phi-3 and GLM stack gate above up.
    pub fn placement(&self, target: Target, layer: usize, config: &Config) -> Option<Placement> {
        let feed_forward = matches!(target, Target::Gate | Target::Up | Target::Down);
        if feed_forward && self.experts.is_some() {
            return None;
        }
        let names = self.names;
        let in_layer = |leaf: &str| format!("{}.{layer}.{leaf}.weight", names.layers);
        let in_attention = |leaf: &str| in_layer(&format!("{}.{leaf}", names.attention));
        let head_dim = self.head_dim;
        let query = self.attention_width(config.num_attention_heads);
        let key_value = config.num_key_value_heads * head_dim;
        let attention_offset = match target {
            Target::Query => Some(0),
            Target::Key => Some(1),
            Target::Value => Some(2),
            _ => None,
        };
        if let Some(slot) = attention_offset {
            return Some(match self.qkv_layout {
                QkvLayout::Separate => Placement::whole(in_attention(match target {
                    Target::Query => names.query,
                    Target::Key => names.key,
                    _ => names.value,
                })),
                QkvLayout::Stacked => {
                    let (start, rows) = match slot {
                        0 => (0, query),
                        1 => (query, key_value),
                        _ => (query + key_value, key_value),
                    };
                    Placement::blocks(in_attention(names.fused_qkv), vec![(start, 0, rows)])
                }
                QkvLayout::HeadInterleaved => Placement::blocks(
                    in_attention(names.fused_qkv),
                    (0..config.num_attention_heads)
                        .map(|head| (head * 3 * head_dim + slot * head_dim, head * head_dim, head_dim))
                        .collect(),
                ),
            });
        }
        let intermediate = config.intermediate_size;
        Some(match target {
            Target::Output => Placement::whole(in_layer(names.output)),
            Target::Gate if self.fused_feed_forward => {
                Placement::blocks(in_layer(names.fused_gate_up), vec![(0, 0, intermediate)])
            }
            Target::Up if self.fused_feed_forward => Placement::blocks(
                in_layer(names.fused_gate_up),
                vec![(intermediate, 0, intermediate)],
            ),
            Target::Gate => Placement::whole(in_layer(names.gate?)),
            Target::Up => Placement::whole(in_layer(names.up)),
            _ => Placement::whole(in_layer(names.down)),
        })
    }

    /// Refuses adapter targets this family has no single projection for, so
    /// an adapter is not created and then never trained.
    pub fn check_targets(&self, targets: &[Target], config: &Config) -> Result<()> {
        if let Some(target) = targets
            .iter()
            .find(|target| self.placement(**target, 0, config).is_none())
        {
            let (why, choices) = if self.experts.is_some() {
                ("its feed-forward is a mixture of experts", "query, key, value, output")
            } else {
                ("its feed-forward has no gate", "query, key, value, output, up, down")
            };
            bail!(
                "this model has no single {} projection to adapt because {why}; choose adapter targets among {choices}",
                target.name()
            );
        }
        Ok(())
    }
}

/// Where one projection's weight sits in the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub tensor: String,
    /// `None` when the projection is the whole tensor; otherwise the row
    /// blocks it owns, as `(row in the tensor, row in the projection,
    /// rows)`.
    pub blocks: Option<Vec<(usize, usize, usize)>>,
}

impl Placement {
    fn whole(tensor: String) -> Self {
        Self { tensor, blocks: None }
    }

    fn blocks(tensor: String, blocks: Vec<(usize, usize, usize)>) -> Self {
        Self {
            tensor,
            blocks: Some(blocks),
        }
    }
}

/// How a checkpoint stores the query, key and value projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvLayout {
    /// Three tensors.
    Separate,
    /// One tensor, all query rows, then key, then value (Phi-3).
    Stacked,
    /// One tensor, head by head: each head's query, key and value rows in
    /// turn (GPT-NeoX).
    HeadInterleaved,
}

/// A routed feed-forward: how many experts, how many run per token, and
/// where the checkpoint keeps them.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureOfExperts {
    pub count: usize,
    pub top_k: usize,
    /// Each expert's inner width.
    pub intermediate: usize,
    /// Rescale the chosen experts' weights to sum to one.
    pub normalize: bool,
    /// Qwen2-MoE's shared expert, by its inner width.
    pub shared_intermediate: Option<usize>,
    pub layout: ExpertLayout,
    /// Bit `i` set means layer `i` keeps a dense feed-forward (Qwen's
    /// `mlp_only_layers` and `decoder_sparse_step`).
    pub dense_layers: u128,
}

/// Where a checkpoint keeps its router and experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLayout {
    /// `block_sparse_moe.gate`, `block_sparse_moe.experts.{e}.w1|w3|w2`.
    Mixtral,
    /// `mlp.gate`, `mlp.experts.{e}.gate_proj|up_proj|down_proj`.
    Qwen,
    /// `block_sparse_moe.router.layer`, and every expert stacked in
    /// `block_sparse_moe.input_linear` and `output_linear`.
    Granite,
}

/// Where a family keeps its tensors. Paths without a dot prefix are below
/// one layer (`{layers}.{i}`); `embeddings`, `layers`, `final_norm` and
/// `lm_head` are from the checkpoint root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Names {
    pub embeddings: &'static str,
    pub layers: &'static str,
    pub final_norm: &'static str,
    pub lm_head: &'static str,
    /// The attention block, below a layer.
    pub attention: &'static str,
    /// The projections, below the attention block.
    pub query: &'static str,
    pub key: &'static str,
    pub value: &'static str,
    pub fused_qkv: &'static str,
    /// The attention output projection.
    pub output: &'static str,
    /// The feed-forward gate, absent from a plain feed-forward.
    pub gate: Option<&'static str>,
    pub up: &'static str,
    pub down: &'static str,
    pub fused_gate_up: &'static str,
    /// The norm before attention (before both halves of a one-norm parallel
    /// block).
    pub attention_norm: &'static str,
    /// The norm over attention's output, in a block that has one.
    pub attention_output_norm: &'static str,
    /// The norm before the feed-forward.
    pub feed_forward_norm: &'static str,
    /// The norm over the feed-forward's output, in a block that has one.
    pub feed_forward_output_norm: &'static str,
}

impl Names {
    /// Llama's names; OLMo 2's post-norm block reuses
    /// `post_attention_layernorm` for the norm over attention's output.
    pub const LLAMA: Self = Self {
        embeddings: "model.embed_tokens",
        layers: "model.layers",
        final_norm: "model.norm",
        lm_head: "lm_head",
        attention: "self_attn",
        query: "q_proj",
        key: "k_proj",
        value: "v_proj",
        fused_qkv: "qkv_proj",
        output: "self_attn.o_proj",
        gate: Some("mlp.gate_proj"),
        up: "mlp.up_proj",
        down: "mlp.down_proj",
        fused_gate_up: "mlp.gate_up_proj",
        attention_norm: "input_layernorm",
        attention_output_norm: "post_attention_layernorm",
        feed_forward_norm: "post_attention_layernorm",
        feed_forward_output_norm: "post_feedforward_layernorm",
    };
    /// Gemma 2 and 3: `post_attention_layernorm` is over attention's output,
    /// so the norm before the feed-forward is `pre_feedforward_layernorm`.
    pub const GEMMA2: Self = Self {
        feed_forward_norm: "pre_feedforward_layernorm",
        ..Self::LLAMA
    };
    /// GLM-4-0414: `post_self_attn_layernorm` over attention's output and
    /// `post_mlp_layernorm` over the feed-forward's.
    pub const GLM4: Self = Self {
        attention_output_norm: "post_self_attn_layernorm",
        feed_forward_output_norm: "post_mlp_layernorm",
        ..Self::LLAMA
    };
    /// Starcoder2's plain feed-forward.
    pub const STARCODER2: Self = Self {
        gate: None,
        up: "mlp.c_fc",
        down: "mlp.c_proj",
        ..Self::LLAMA
    };
    /// A plain feed-forward that keeps Llama's up and down names (Nemotron,
    /// Arcee).
    pub const UP_DOWN: Self = Self {
        gate: None,
        ..Self::LLAMA
    };
    /// Phi-2: `dense` for the attention output, `fc1`/`fc2`, and a
    /// `final_layernorm`.
    pub const PHI: Self = Self {
        output: "self_attn.dense",
        gate: None,
        up: "mlp.fc1",
        down: "mlp.fc2",
        final_norm: "model.final_layernorm",
        ..Self::LLAMA
    };
    /// GPT-NeoX and Pythia: everything below `gpt_neox`, one head-interleaved
    /// `query_key_value`, `dense_h_to_4h`/`dense_4h_to_h`, and `embed_out`.
    pub const GPT_NEOX: Self = Self {
        embeddings: "gpt_neox.embed_in",
        layers: "gpt_neox.layers",
        final_norm: "gpt_neox.final_layer_norm",
        lm_head: "embed_out",
        attention: "attention",
        fused_qkv: "query_key_value",
        output: "attention.dense",
        gate: None,
        up: "mlp.dense_h_to_4h",
        down: "mlp.dense_4h_to_h",
        ..Self::LLAMA
    };
    /// GPT-J: everything below `transformer`, `h.{i}`, `ln_1`, `attn`,
    /// `out_proj`, `fc_in`/`fc_out`, and `ln_f`.
    pub const GPT_J: Self = Self {
        embeddings: "transformer.wte",
        layers: "transformer.h",
        final_norm: "transformer.ln_f",
        attention: "attn",
        output: "attn.out_proj",
        gate: None,
        up: "mlp.fc_in",
        down: "mlp.fc_out",
        attention_norm: "ln_1",
        ..Self::LLAMA
    };
}

/// Whether the feed-forward gates its up projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardKind {
    /// `down(act(gate(x)) * up(x))`.
    Gated,
    /// `down(act(up(x)))`.
    Plain,
}

/// How a family normalises, read from its config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Root mean square, no mean subtraction, no bias.
    Rms,
    /// Mean subtracted, then divided by the standard deviation; `bias` says
    /// whether a bias tensor follows the scale.
    Layer { bias: bool },
    /// LayerNorm with no scale and no bias at all (OLMo 1).
    Bare,
}

/// A rotary scaling read from the config's `rope_scaling`, beyond Llama 3's.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    None,
    /// Every angle divided by the factor.
    Linear(f32),
    /// Phi-3's LongRoPE: each frequency divided by its own factor, from
    /// `short` while the sequence is within `original` positions and from
    /// `long` once it goes beyond, and both tables multiplied by `attention`.
    LongRope {
        short: Vec<f32>,
        long: Vec<f32>,
        original: usize,
        attention: f32,
    },
}

/// Where a query and key norm sits, if the family has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKeyNorm {
    None,
    /// One norm over each head's `head_dim`, shared by every head (Qwen3,
    /// Gemma 3).
    PerHead,
    /// One norm over the whole projection (OLMo 2).
    Full,
    /// A separate scale per head, stored as one `[heads, head_dim]` tensor
    /// (Cohere's `use_qk_norm`).
    HeadWeights,
    /// A separate norm per head, stored as one module per head (StableLM's
    /// `qk_layernorm`).
    HeadModules,
}

/// The non-linearity in the feed-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    /// `gelu_pytorch_tanh` and `gelu_new`, which Candle's `gelu` computes.
    GeluTanh,
    /// The exact, erf-based GELU.
    Gelu,
    /// Squared ReLU (Nemotron's `relu2`).
    Relu2,
}

impl Activation {
    pub(crate) fn apply(self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Silu => candle_nn::ops::silu(input),
            Self::GeluTanh => input.gelu(),
            Self::Gelu => input.gelu_erf(),
            Self::Relu2 => input.relu()?.sqr(),
        }
    }
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
