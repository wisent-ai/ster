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
//! * **Rotary is F32 in and F32 through, and casts back.** The angles are
//!   computed F32 whatever the weights are, because they come from absolute
//!   position rather than from a weight: in F16 two neighbouring late
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

pub(crate) use attention::ALIBI_SPAN;
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
    /// Normalise each head's query and key after the rotation (HunYuan)
    /// rather than before it.
    pub norm_after_rotary: bool,
    /// The layers whose softmax holds a learned sink logit per head
    /// (GPT-OSS's `self_attn.sinks` on every layer, MiMo-V2's
    /// `attention_sink_bias` on its sliding-window layers).
    pub attention_sinks: u128,
    /// `clip_qkv`: every query, key and value component clamped to this
    /// bound (OLMo, OLMoE, DBRX).
    pub clip_qkv: Option<f64>,
    pub query_key_value_bias: bool,
    pub output_bias: bool,
    /// Bias on the feed-forward projections (Starcoder2, Phi-2, Nemotron's
    /// `mlp_bias`).
    pub feed_forward_bias: bool,
    /// A bias on the down projection even where the gate and up have none
    /// (TeleChat2).
    pub down_bias: bool,
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
    /// When set, the windowed layers attend by chunks rather than through a
    /// sliding window: `sliding_window` is the chunk size, and a query sees
    /// its own chunk and this many before it (Llama 4's chunked attention,
    /// none; Rnj-1's, one).
    pub chunk_lookback: Option<usize>,
    /// Bit `i` set means layer `i` applies no rotary embedding.
    pub unrotated_layers: u128,
    /// The rotary base sliding-window layers use, when it differs from
    /// `rope_theta` (Gemma 3).
    pub local_rope_theta: Option<f32>,
    /// A rotary scaling Ster applies itself; Llama 3's is carried by Candle's
    /// config instead.
    pub rope_scaling: RopeScaling,
    pub qkv_layout: QkvLayout,
    /// How a token's position enters the model.
    pub positions: Positions,
    /// Projection weights stored `[inputs, outputs]`, as GPT-2's `Conv1D`
    /// stores them, rather than `[outputs, inputs]`.
    pub conv1d: bool,
    /// A norm straight after the embedding (BLOOM).
    pub embedding_norm: bool,
    /// Gate and up stored as one `gate_up_proj`, gate rows first (Phi-3,
    /// GLM).
    pub fused_feed_forward: bool,
    pub norm_offset: bool,
    /// How many equal groups every hidden-width norm normalises apart
    /// (K2-Horizon's `layernorm_num_groups`); one for a plain norm.
    pub norm_groups: usize,
    /// What the embedding is multiplied by before the first block.
    pub embedding_multiplier: Option<f64>,
    /// What each sublayer's output is multiplied by before the residual add.
    pub residual_multiplier: Option<f64>,
    /// Nemotron-H: every block is one norm and one sublayer. The layers in
    /// this set run the feed-forward alone; its attention layers have no
    /// feed-forward.
    pub lone_sublayers: Option<u128>,
    /// Falcon-H1: every layer runs attention and a Mamba-2 scan on the same
    /// normed input and adds both, each scaled, before the feed-forward.
    pub parallel_scan: Option<ParallelScan>,
    /// Falcon-H1's `key_multiplier` on every key.
    pub key_scale: Option<f64>,
    /// MiMo-V2's `attention_value_scale` on every value.
    pub value_scale: Option<f64>,
    /// Llama 4's query temperature on the layers it names (Llama 4's
    /// NoPE layers, every layer of Ministral 3).
    pub query_temperature: Option<QueryTemperature>,
    /// Each value head's width when it differs from the query and key
    /// heads' (MiMo-V2's `v_head_dim`).
    pub value_head_dim: Option<usize>,
    /// Falcon-H1's `mlp_multipliers`: the gate's pre-activation and the
    /// feed-forward's output are multiplied by these.
    pub feed_forward_scales: Option<(f64, f64)>,
    /// Qwen3-Next's gated delta-rule mixers, in place of attention on the
    /// layers they name.
    pub delta_rule: Option<DeltaRuleSpec>,
    /// Qwen3-Next's attention gate: `q_proj` yields each head's query rows
    /// and then as many gate rows, and attention's output is multiplied by
    /// the gate's sigmoid before `o_proj`.
    pub output_gate: bool,
    /// A head-wise gate: `g_proj` yields one logit per head, and each head's
    /// output is multiplied by it through this function before `o_proj`
    /// (Step 3.5's `use_head_wise_attn_gate`, a sigmoid; Laguna's `gating`
    /// `per-head`, a softplus).
    pub head_gate: Option<GateFunction>,
    /// K2-Horizon's elementwise attention gate (`attention_gate_func`):
    /// `gate_proj` of the hidden state, through this function, multiplies
    /// attention's output before `o_proj`.
    pub attention_gate: Option<GateFunction>,
    /// K2-Horizon's mixture of value experts (MoVA) on its routed layers: no
    /// `v_proj`, each value the router-weighted sum of its chosen
    /// `v_experts`' SiLU outputs.
    pub value_experts: Option<MixtureOfExperts>,
    /// Step3's query bottleneck width (`share_q_dim`): the query is
    /// `wq(inter_norm(q_proj(x)))`.
    pub query_bottleneck: Option<usize>,
    /// MiniMax-Text-01's lightning attention, in place of attention on the
    /// layers it names.
    pub lightning: Option<LightningSpec>,
    /// MiniMax-Text-01's scaled residuals: each sublayer's output joins the
    /// residual as `residual · alpha + output · beta`.
    pub scaled_residuals: Option<ScaledResiduals>,
    /// Zamba2's shared transformer blocks.
    pub shared_blocks: Option<SharedBlocksSpec>,
    /// Gemma 4's full-attention layers: wider heads, their own key-value
    /// head count, and values read from the key projection.
    pub global_attention: Option<GlobalAttention>,
    /// A scale-free RMS norm over each head's value (Gemma 4's `v_norm`).
    pub value_norm: bool,
    /// Gemma 4's key-value sharing: from this layer on, a layer projects
    /// only its query and reuses the keys and values of the last earlier
    /// layer of its own kind (`num_kv_shared_layers`).
    pub shared_key_values: Option<usize>,
    /// Gemma 4's per-layer inputs (`hidden_size_per_layer_input`).
    pub per_layer_input: Option<PerLayerInputSpec>,
    /// LongCat-Flash-Lite's hashed n-gram embeddings beside the token
    /// embedding.
    pub ngram: Option<NgramSpec>,
    /// Gemma 4's experts beside the dense feed-forward on every layer: the
    /// router reads the residual through a scale-free norm, the experts read
    /// it through `pre_feedforward_layernorm_2`, and the two outputs, each
    /// normed, are added.
    pub side_experts: Option<MixtureOfExperts>,
    /// Every block's output multiplied by its stored `layer_scalar`
    /// (Gemma 4).
    pub layer_scalar: bool,
    /// Solar's block skip connections.
    pub skip_connections: Option<SkipConnections>,
    /// Nanbeige's loops: the stored layers run more than once.
    pub loops: Option<Loops>,
    /// DeciLM's per-layer plan, one entry per layer.
    pub layer_plans: Option<Vec<LayerPlan>>,
    /// DeepSeek Sparse Attention's indexer (DeepSeek-V3.2, GLM-5).
    pub sparse_index: Option<IndexerSpec>,
    /// A norm over each routed layer's feed-forward output, named by
    /// `names.feed_forward_output_norm` (A.X-K1's `post_mlp_layernorm`).
    pub routed_output_norm: bool,
    pub activation: Activation,
    /// A norm before attention and before the feed-forward (every family but
    /// OLMo 2).
    pub pre_norms: bool,
    /// A norm over attention's and the feed-forward's outputs (Gemma 2 and 3,
    /// OLMo 2).
    pub output_norms: bool,
    /// The output norms' own epsilon where it differs from the other norms'
    /// (MuseGlimmer's `post_norm_eps`).
    pub output_norm_eps: Option<f64>,
    /// The final norm stores its scale as is while the block norms store an
    /// offset from one (MuseGlimmer).
    pub plain_final_norm: bool,
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
    /// Each layer's own `(top_k, intermediate)` where it differs from
    /// `experts` (Nemotron Puzzle's `block_configs`).
    pub expert_overrides: Option<Vec<Option<(usize, usize)>>>,
    /// DeepSeek's multi-head latent attention in place of separate query,
    /// key and value projections.
    pub latent: Option<LatentAttention>,
    /// LongCat-Flash's `mla_scale_q_lora` and `mla_scale_kv_lora`: the
    /// latent query and key-value bottlenecks' normed outputs multiplied by
    /// these, `(query, key_value)`.
    pub latent_scales: Option<(f64, f64)>,
    /// LongCat-Flash's shortcut-connected experts: each stored layer is two
    /// Ster layers (attention and feed-forward each), and the experts read
    /// the first half's feed-forward input and join the residual at the end
    /// of the second half.
    pub shortcut_experts: Option<MixtureOfExperts>,
    /// Mamba's selective state-space mixer in place of attention, with no
    /// feed-forward beside it.
    pub state_space: Option<StateSpaceSpec>,
    /// LFM2's gated short convolution in place of attention on the layers it
    /// covers.
    pub short_convolution: Option<ShortConvolution>,
}

/// DeepSeek Sparse Attention's indexer: `heads` heads of `head_dim`
/// (`index_n_heads`, `index_head_dim`), keeping each query's `top_k`
/// (`index_topk`) keys, rotating adjacent pairs under `interleaved` (GLM-5's
/// `indexer_rope_interleave`) and halves otherwise; the layers in
/// `shared_layers` (GLM-5's `indexer_types` `shared`) have no indexer and
/// reuse the last indexed layer's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub interleaved: bool,
    pub shared_layers: u128,
}

impl IndexerSpec {
    pub fn shared(&self, layer: usize) -> bool {
        layer < u128::BITS as usize && self.shared_layers & (1u128 << layer) != 0
    }
}

/// One DeciLM layer (`block_configs`): its key-value heads, `None` for a
/// no-op attention, and its feed-forward width, `None` for a no-op
/// feed-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPlan {
    pub key_value_heads: Option<usize>,
    pub intermediate: Option<usize>,
}

/// A looped decoder: `physical` stored layers run in order as many times as
/// the config's layer count holds them, each pass with its own key-value
/// cache slots. Under `norm_between` (Nanbeige's `num_loops`) the final
/// norm also closes every pass but the last; with `gate_window`
/// (IQuest-LoopCoder's `loop_window_size`) every later pass mixes global
/// attention over the first pass's keys with local attention over its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loops {
    pub physical: usize,
    pub count: usize,
    pub norm_between: bool,
    pub gate_window: Option<usize>,
}

/// Solar's block skip connections (`bskcn_1` to `bskcn_4`): before the
/// layers in `save[i]` the hidden state is kept in slot `i`; before the
/// layers in `blend[i]` it becomes `weight · kept[i] + (1 − weight) · hidden`,
/// `weight` being the inference value of `bskcn_tv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkipConnections {
    pub save: [u128; 2],
    pub blend: [u128; 2],
    pub weight: f64,
}

impl SkipConnections {
    /// The hidden state layer `layer` reads, keeping what it must in `kept`.
    pub(crate) fn apply(
        &self,
        layer: usize,
        hidden: Tensor,
        kept: &mut [Option<Tensor>; 2],
    ) -> candle_core::Result<Tensor> {
        let listed = |set: u128| layer < u128::BITS as usize && set & (1u128 << layer) != 0;
        for slot in 0..2 {
            if listed(self.save[slot]) {
                kept[slot] = Some(hidden.clone());
            }
        }
        let mut hidden = hidden;
        for slot in 0..2 {
            if listed(self.blend[slot]) {
                let Some(earlier) = &kept[slot] else {
                    candle_core::bail!("layer {layer} blends in a hidden state no earlier layer kept");
                };
                hidden = ((earlier * self.weight)? + (hidden * (1.0 - self.weight))?)?;
            }
        }
        Ok(hidden)
    }
}

/// The full-attention layers' own attention shape, beside the
/// sliding-window layers': `heads` query heads when the config states them
/// apart (Step 3.5's `num_attention_heads` beside
/// `attention_other_setting`), heads `head_dim` wide (Gemma 4's
/// `global_head_dim`), `key_value_heads` of them when the config states
/// them apart (Gemma 4's `num_global_key_value_heads`, MiMo-V2's
/// `num_key_value_heads` beside `swa_num_key_value_heads`), the first
/// `rotary_dim` components of each rotated by the global table, and, under
/// `key_is_value` (Gemma 4's `attention_k_eq_v`), values taken from the
/// key projection before its norm, with no `v_proj`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalAttention {
    pub heads: Option<usize>,
    pub head_dim: usize,
    pub key_value_heads: Option<usize>,
    pub rotary_dim: usize,
    pub key_is_value: bool,
}

/// Gemma 4's per-layer inputs: `embed_tokens_per_layer` (`vocab` rows of
/// `width` per layer, multiplied by `sqrt(width)`) beside
/// `per_layer_model_projection` of the scaled embeddings (times
/// `hidden_size^-0.5`, then `per_layer_projection_norm`), the two added and
/// multiplied by `2^-0.5`. Each layer gates its slice with
/// `per_layer_input_gate`, projects it back with `per_layer_projection`,
/// normalises it with `post_per_layer_input_norm` and adds it to its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerLayerInputSpec {
    pub width: usize,
    pub vocab: usize,
}

/// LongCat's n-gram embeddings (`ngram_vocab_size_ratio`, `emb_split_num`,
/// `emb_neighbor_num`): for each n-gram order from 2 to `neighbors` and each
/// of `splits` tables, the token and the ones before it hash to a row of
/// table `(order - 2) · splits + split`, `ratio · vocab + 2 · index + 1`
/// rows tall; each row is projected to the hidden width, and the token
/// embedding and every projection are averaged. A run of tokens restarts
/// after each `eos` token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramSpec {
    pub ratio: usize,
    pub splits: usize,
    pub neighbors: usize,
    pub eos: u32,
}

/// LFM2's gated short convolution: `in_proj` to `B`, `C` and `x`, a causal
/// depthwise convolution of `B·x` over `kernel` positions, `C` times its
/// output, and `out_proj`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvolution {
    /// `conv_L_cache`: how many positions the convolution spans.
    pub kernel: usize,
    /// `conv_bias`: bias on both projections and the convolution.
    pub bias: bool,
    /// Bit `i` set means layer `i` convolves rather than attends.
    pub layers: u128,
}

/// Mamba's mixer dimensions, from `intermediate_size` (or `expand` times the
/// width), `state_size`, `conv_kernel` and `time_step_rank`, and which layers
/// use it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateSpaceSpec {
    pub inner: usize,
    pub state: usize,
    pub kernel: usize,
    pub step_rank: usize,
    /// `use_bias`: bias on `in_proj` and `out_proj`.
    pub projection_bias: bool,
    /// `use_conv_bias`.
    pub convolution_bias: bool,
    /// A norm on the step and the input and output matrices.
    pub parameter_norm: ParameterNorm,
    /// Bit `i` set means layer `i` mixes with the state-space model rather
    /// than attention: every layer in Mamba, the non-attention ones in Jamba.
    pub layers: u128,
    /// A feed-forward follows the mixer (Jamba); Mamba's block is the mixer
    /// alone.
    pub feed_forward: bool,
    /// Mamba-2's structured mixer in place of Mamba-1's selective one.
    pub structured: Option<StructuredSpec>,
}

/// Mamba-2's multi-head structured state-space mixer (the SSD layer): the
/// inner width splits into `heads` heads of `head_dim`, each with one scalar
/// decay, and the input and output matrices are shared within `groups`
/// groups of heads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    /// `time_step_limit`: the step is clamped to this range after softplus.
    pub step_limit: (f64, f64),
    /// Falcon-H1's `ssm_in_multiplier` on the mixer's input (one elsewhere).
    pub input_scale: f64,
    /// Falcon-H1's `ssm_multipliers` on the projection's five parts — gate,
    /// stream, input matrix, output matrix and step — in that order.
    pub projection_scales: Option<[f64; 5]>,
    /// Whether the gated output passes the group RMS norm (`mamba.norm`);
    /// Falcon-H1's `mamba_rms_norm` false leaves it gated only.
    pub gated_norm: bool,
}

/// A gated delta-rule linear attention: `key_heads` query and key heads of
/// `key_dim`, `value_heads` value heads of `value_dim` (each key head serving
/// `value_heads / key_heads` of them), a causal depthwise convolution of
/// `kernel` positions over query, key and value, on the layers in `layers`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaRuleSpec {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
    pub layers: u128,
    pub form: DeltaRuleForm,
    /// OLMo Hybrid's `linear_allow_neg_eigval`: the write strength is
    /// `2 · sigmoid(b)`, so the state's update may reflect as well as shrink.
    pub negative_eigenvalues: bool,
    /// Kimi Delta Attention's safe gate (Ling 3.0's `kda_safe_gate`): the
    /// log-decay is `floor · sigmoid(exp(A_log) · (input + dt_bias))`, never
    /// below this floor, in place of `-exp(A_log) · softplus(input +
    /// dt_bias)`.
    pub decay_floor: Option<f64>,
}

/// Which family's delta rule a layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaRuleForm {
    /// Qwen3-Next's Gated DeltaNet (`linear_attn`): fused `in_proj_qkvz` and
    /// `in_proj_ba`, one decay per value head, a SiLU output gate.
    Qwen3Next,
    /// Qwen3.5's Gated DeltaNet (`linear_attn`): Qwen3-Next's, with
    /// `in_proj_qkv` holding every query, then every key, then every value,
    /// and the gate, `b` and decay input in `in_proj_z`, `in_proj_b` and
    /// `in_proj_a`.
    Qwen35,
    /// Kimi Delta Attention (`self_attn`): separate `q_proj`, `k_proj` and
    /// `v_proj` with their own convolutions, a decay per key channel from
    /// the low-rank `f_a_proj`/`f_b_proj`, and a sigmoid output gate from
    /// `g_a_proj`/`g_b_proj`.
    Kimi,
    /// OLMo Hybrid's Gated DeltaNet (`linear_attn`): separate `q_proj`,
    /// `k_proj` and `v_proj` under one `conv1d`, `a_proj` and `b_proj` one
    /// per value head, a SiLU output gate from `g_proj`, `o_norm` and
    /// `o_proj`.
    OlmoHybrid,
    /// Ling 3.0's Kimi Delta Attention (`attention`): Kimi's, with the decay
    /// input and the output gate from full-rank `f_proj` and `g_proj`.
    Ling,
}

/// Zamba2's shared transformer blocks: on every layer in `hybrid_layers`, a
/// block reads the hidden state beside the embeddings (`attention_input`
/// wide), and its output, projected back by the layer's own `linear`, joins
/// the input of the layer's Mamba-2 mixer. There are `blocks` distinct
/// blocks, used in turn; the k-th hybrid layer adds its own rank-`rank`
/// terms (slot k of the block's adapter lists) to the block's query, key
/// and value when `attention_adapters`, and to its fused gate and up
/// projection, `intermediate` wide each, when `feed_forward_adapters`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedBlocksSpec {
    pub hybrid_layers: u128,
    pub blocks: usize,
    pub attention_input: usize,
    pub intermediate: usize,
    pub rank: usize,
    pub attention_adapters: bool,
    pub feed_forward_adapters: bool,
}

impl SharedBlocksSpec {
    /// The hybrid layers, in order.
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        (0..u128::BITS as usize).filter(|layer| self.hybrid_layers & (1u128 << layer) != 0)
    }

    /// How many hybrid layers come before `layer`: its slot in the adapter
    /// lists, and its block's position in the rotation.
    pub fn slot(&self, layer: usize) -> usize {
        self.layers().take_while(|hybrid| *hybrid < layer).count()
    }
}

/// Lightning attention: `heads` heads of `head_dim`, on the layers in
/// `layers`, in MiniMax-Text-01's or Ling 2.5's form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightningSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub layers: u128,
    pub form: LightningForm,
}

/// Which family's lightning attention a layer runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightningForm {
    /// MiniMax-Text-01 (`self_attn`): `qkv_proj` through SiLU, one RMS
    /// norm over all heads, a sigmoid `output_gate`, `out_proj`.
    MiniMax,
    /// Ling 2.5 (`attention`): `query_key_value` (through SiLU under
    /// `linear_silu`), per-head `query_layernorm` and `key_layernorm` under
    /// `qk_norm`, the global rotation by halves, the query scaled by
    /// `head_dim^-0.5`, an RMS norm over `groups` equal groups of the
    /// read-out scaled by `g_norm`, a sigmoid `g_proj` gate, `dense`.
    Bailing { groups: usize, silu: bool, qk_norm: bool },
}

/// MiniMax-Text-01's residual scaling: `(alpha, beta)` for the lightning
/// attention layers, the full attention layers and the feed-forward. With
/// `from_normed` (`postnorm`), the residual each sublayer joins is its
/// normed input rather than the stream before the norm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledResiduals {
    pub from_normed: bool,
    pub linear_attention: (f64, f64),
    pub full_attention: (f64, f64),
    pub feed_forward: (f64, f64),
}

/// Falcon-H1's parallel block: attention reads its input times
/// `attention_in`, and the attention and scan outputs are multiplied by
/// `attention_out` and `scan_out` before they join the residual stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallelScan {
    pub attention_in: f64,
    pub attention_out: f64,
    pub scan_out: f64,
}

/// The norm on a state-space mixer's step and input and output matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterNorm {
    None,
    /// Falcon-Mamba's `mixer_rms_eps`: an RMS norm with no scale.
    Bare(f64),
    /// Jamba's `dt_layernorm`, `b_layernorm` and `c_layernorm`: RMS norms
    /// with a stored scale.
    Weighted(f64),
}

/// Multi-head latent attention (DeepSeek-V2 and V3, MiniCPM3): query and key
/// value pass through low-rank bottlenecks, each head's key is a
/// position-free part from the key-value bottleneck plus one rotated part
/// shared by every head, and a head's value may be narrower than its query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentAttention {
    /// `q_lora_rank`: the query bottleneck, or `None` for a direct `q_proj`.
    pub query_rank: Option<usize>,
    /// `kv_lora_rank`: the key-value bottleneck.
    pub key_value_rank: usize,
    /// `qk_nope_head_dim`: each head's position-free query and key part.
    pub unrotated: usize,
    /// `qk_rope_head_dim`: each head's rotated part.
    pub rotated: usize,
    /// `v_head_dim`.
    pub value: usize,
}

impl Architecture {
    /// The plain Llama block with a head width derived from the residual one.
    pub fn llama(hidden_size: usize, heads: usize, norm_eps: f64) -> Self {
        let head_dim = hidden_size / heads;
        Self {
            head_dim,
            rotary_dim: head_dim,
            query_key_norm: QueryKeyNorm::None,
            norm_after_rotary: false,
            attention_sinks: 0,
            clip_qkv: None,
            query_key_value_bias: false,
            output_bias: false,
            feed_forward_bias: false,
            down_bias: false,
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
            chunk_lookback: None,
            unrotated_layers: 0,
            local_rope_theta: None,
            rope_scaling: RopeScaling::None,
            qkv_layout: QkvLayout::Separate,
            positions: Positions::Rotary,
            conv1d: false,
            embedding_norm: false,
            fused_feed_forward: false,
            norm_offset: false,
            norm_groups: 1,
            embedding_multiplier: None,
            residual_multiplier: None,
            lone_sublayers: None,
            parallel_scan: None,
            key_scale: None,
            value_scale: None,
            query_temperature: None,
            value_head_dim: None,
            feed_forward_scales: None,
            delta_rule: None,
            output_gate: false,
            head_gate: None,
            attention_gate: None,
            value_experts: None,
            query_bottleneck: None,
            lightning: None,
            scaled_residuals: None,
            shared_blocks: None,
            global_attention: None,
            value_norm: false,
            shared_key_values: None,
            per_layer_input: None,
            ngram: None,
            side_experts: None,
            layer_scalar: false,
            skip_connections: None,
            loops: None,
            layer_plans: None,
            sparse_index: None,
            routed_output_norm: false,
            activation: Activation::Silu,
            pre_norms: true,
            output_norms: false,
            output_norm_eps: None,
            plain_final_norm: false,
            score_divisor: (head_dim as f64).sqrt(),
            attention_softcap: None,
            final_softcap: None,
            logits_multiplier: None,
            experts: None,
            expert_overrides: None,
            latent: None,
            latent_scales: None,
            shortcut_experts: None,
            state_space: None,
            short_convolution: None,
        }
    }

    /// The window layer `layer` attends through, or `None` for full attention.
    pub fn window(&self, layer: usize) -> Option<usize> {
        let sliding = layer < 128 && self.sliding_layers & (1u128 << layer) != 0;
        self.sliding_window.filter(|_| sliding)
    }

    /// The stored layer Ster layer `layer` reads its tensors from, and the
    /// names it reads them by: the half of a LongCat-Flash layer
    /// (`self_attn.0`, `mlps.0`, … then `.1`), or the layer itself.
    pub fn stored_layer(&self, layer: usize) -> (usize, Names) {
        match self.shortcut_experts {
            Some(_) if layer % 2 == 0 => (layer / 2, Names::LONGCAT_FIRST),
            Some(_) => (layer / 2, Names::LONGCAT_SECOND),
            None => (layer, self.names),
        }
    }

    /// DeciLM's plan for `layer`, when the family states one per layer.
    pub fn layer_plan(&self, layer: usize) -> Option<LayerPlan> {
        self.layer_plans.as_ref().and_then(|plans| plans.get(layer).copied())
    }

    /// The full-attention spec layer `layer` uses: Gemma 4's or MiMo-V2's
    /// on a layer that attends past any window, `None` otherwise.
    pub fn global_at(&self, layer: usize) -> Option<GlobalAttention> {
        self.global_attention.filter(|_| self.window(layer).is_none())
    }

    /// Whether layer `layer`'s softmax holds a learned sink logit per head.
    pub fn sinks_at(&self, layer: usize) -> bool {
        layer < 128 && self.attention_sinks & (1u128 << layer) != 0
    }

    /// How many components the global rotation spans: the full-attention
    /// layers' own rotated width under a global attention spec,
    /// `rotary_dim` otherwise.
    pub fn global_rotary_dim(&self) -> usize {
        self.global_attention.map_or(self.rotary_dim, |global| global.rotary_dim)
    }

    /// The layer whose keys and values a Gemma 4 key-value-sharing layer
    /// reuses: the last layer before the sharing ones whose attention is of
    /// the same kind (sliding-window or full).
    pub fn key_value_source(&self, layer: usize) -> Option<usize> {
        let first = self.shared_key_values.filter(|first| layer >= *first)?;
        let kind = self.window(layer).is_some();
        (0..first).rev().find(|earlier| self.window(*earlier).is_some() == kind)
    }

    /// Whether `layer` is the one whose keys and values the sharing layers of
    /// its kind reuse.
    pub fn stores_key_values(&self, layer: usize) -> bool {
        let Some(first) = self.shared_key_values.filter(|first| layer < *first) else {
            return false;
        };
        let kind = self.window(layer).is_some();
        (layer + 1..first).all(|later| self.window(later).is_some() != kind)
    }

    /// Whether layer `layer` applies the rotary embedding: never for a family
    /// whose positions are learned or ALiBi.
    pub fn rotates(&self, layer: usize) -> bool {
        self.positions == Positions::Rotary
            && (layer >= 128 || self.unrotated_layers & (1u128 << layer) == 0)
    }

    /// Width of the query projection and of the attention output.
    pub fn attention_width(&self, heads: usize) -> usize {
        heads * self.head_dim
    }

    /// The short convolution layer `layer` uses in place of attention, if
    /// any.
    pub fn short_convolution_at(&self, layer: usize) -> Option<&ShortConvolution> {
        self.short_convolution
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// The state-space mixer layer `layer` uses in place of attention, if
    /// any.
    pub fn state_space_at(&self, layer: usize) -> Option<&StateSpaceSpec> {
        self.state_space
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// The lightning attention layer `layer` uses in place of attention, if
    /// any.
    pub fn lightning_at(&self, layer: usize) -> Option<&LightningSpec> {
        self.lightning
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// The gated delta-rule mixer layer `layer` uses in place of attention,
    /// if any.
    pub fn delta_rule_at(&self, layer: usize) -> Option<&DeltaRuleSpec> {
        self.delta_rule
            .as_ref()
            .filter(|spec| layer < 128 && spec.layers & (1u128 << layer) != 0)
    }

    /// Layer `layer`'s mixture of experts, with the layer's own count of
    /// chosen experts and width where the family states one.
    pub fn experts_at(&self, layer: usize) -> Option<MixtureOfExperts> {
        let mut experts = self.experts.clone()?;
        if let Some(Some((top_k, intermediate))) = self.expert_overrides.as_ref().and_then(|all| all.get(layer)) {
            experts.top_k = *top_k;
            experts.intermediate = *intermediate;
        }
        Some(experts)
    }

    /// Whether layer `layer`'s feed-forward is the mixture of experts.
    pub fn routed(&self, layer: usize) -> bool {
        self.experts.as_ref().is_some_and(|experts| {
            layer >= 128 || experts.dense_layers & (1u128 << layer) == 0
        })
    }

    /// Width of one head's value: narrower than the query in latent
    /// attention and under MiMo-V2's `v_head_dim`, the head width otherwise.
    pub fn value_dim(&self) -> usize {
        self.latent
            .map(|latent| latent.value)
            .or(self.value_head_dim)
            .unwrap_or(self.head_dim)
    }

    /// Where `target`'s weight lives at `layer`, or `None` when this family
    /// has no single such projection: the gate of a plain feed-forward, any
    /// feed-forward projection of a mixture of experts, or the key, value and
    /// bottlenecked query of latent attention.
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
        // A Mamba block has no attention or feed-forward projection.
        if self.state_space.is_some() || self.delta_rule.is_some() || self.lightning.is_some() {
            return None;
        }
        // Step3's query passes a bottleneck and a norm before `wq`, so `wq`
        // reads no hidden state an adapter could share.
        if self.query_bottleneck.is_some() && target == Target::Query {
            return None;
        }
        // Gemma 4's full-attention heads are wider than its sliding-window
        // ones, and MiMo-V2's have fewer key-value heads, so no attention
        // projection has one shape on every layer.
        if self.global_attention.is_some() && !feed_forward {
            return None;
        }
        // K2-Horizon's routed layers take their value from value experts,
        // with no `v_proj`.
        if self.value_experts.is_some() && target == Target::Value {
            return None;
        }
        if let Some(latent) = self.latent {
            match target {
                Target::Query if latent.query_rank.is_none() => {
                    return Some(Placement::whole(in_attention(names.query)));
                }
                Target::Query | Target::Key | Target::Value => return None,
                _ => {}
            }
        }
        let head_dim = self.head_dim;
        let query = self.attention_width(config.num_attention_heads);
        let key_value = config.num_key_value_heads * head_dim;
        let attention_offset = match target {
            Target::Query => Some(0),
            Target::Key => Some(1),
            Target::Value => Some(2),
            _ => None,
        };
        let intermediate = config.intermediate_size;
        let placement = match (attention_offset, self.qkv_layout) {
            // Each head's query rows sit before its gate rows.
            (Some(0), QkvLayout::Separate) if self.output_gate => {
                let blocks = (0..config.num_attention_heads)
                    .map(|head| (head * 2 * head_dim, head * head_dim, head_dim))
                    .collect();
                Placement::blocks(in_attention(names.query), blocks)
            }
            (Some(slot), QkvLayout::Separate) => Placement::whole(in_attention(match slot {
                0 => names.query,
                1 => names.key,
                _ => names.value,
            })),
            (Some(slot), QkvLayout::Stacked) => {
                let (start, rows) = match slot {
                    0 => (0, query),
                    1 => (query, key_value),
                    _ => (query + key_value, key_value),
                };
                Placement::blocks(in_attention(names.fused_qkv), vec![(start, 0, rows)])
            }
            (Some(slot), QkvLayout::Grouped) => {
                // Each key-value group holds its query heads, then its key
                // head, then its value head.
                let groups = config.num_key_value_heads;
                let per_group = config.num_attention_heads / groups;
                let stride = (per_group + 2) * head_dim;
                let blocks = (0..groups)
                    .flat_map(|group| {
                        let (first, count, own) = match slot {
                            0 => (0, per_group, group * per_group),
                            1 => (per_group, 1, group),
                            _ => (per_group + 1, 1, group),
                        };
                        (0..count).map(move |head| {
                            (group * stride + (first + head) * head_dim, (own + head) * head_dim, head_dim)
                        })
                    })
                    .collect();
                Placement::blocks(in_attention(names.fused_qkv), blocks)
            }
            (Some(0), QkvLayout::PairedKeyValue) => Placement::whole(in_attention(names.query)),
            (Some(slot), QkvLayout::PairedKeyValue) => {
                // Each key-value head holds its key rows, then its value rows.
                let first = if slot == 1 { 0 } else { head_dim };
                let blocks = (0..config.num_key_value_heads)
                    .map(|head| (head * 2 * head_dim + first, head * head_dim, head_dim))
                    .collect();
                Placement::blocks(in_attention(names.fused_qkv), blocks)
            }
            (None, _) => match target {
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
            },
        };
        // A multimodal checkpoint keeps the language model below a wrapper
        // (Gemma 3's `language_model`).
        let tensor = if names.wrapper.is_empty() {
            placement.tensor
        } else {
            format!("{}.{}", names.wrapper, placement.tensor)
        };
        Some(Placement {
            tensor,
            transposed: self.conv1d,
            root: names.root,
            ..placement
        })
    }

    /// Refuses adapter targets this family has no single projection for, so
    /// an adapter is not created and then never trained.
    pub fn check_targets(&self, targets: &[Target], config: &Config) -> Result<()> {
        if self.short_convolution.is_some() && !targets.is_empty() {
            bail!(
                "this model's blocks include short-convolution mixers (LFM2) with no attention projection to adapt on every layer; Ster steers it but trains no adapters on it"
            );
        }
        if self.layer_plans.is_some() && !targets.is_empty() {
            bail!(
                "this model's layers differ in key-value heads and feed-forward width, and some have neither (DeciLM's block_configs), so no projection has one shape on every layer; Ster steers it but trains no adapters on it"
            );
        }
        if self.shortcut_experts.is_some() && !targets.is_empty() {
            bail!(
                "this model's layers each hold two attention and feed-forward pairs beside shortcut-connected experts (LongCat-Flash), so no projection sits at one name per Ster layer; Ster steers it but trains no adapters on it"
            );
        }
        if self.loops.is_some() && !targets.is_empty() {
            bail!(
                "this model runs its layers more than once (Nanbeige's num_loops, IQuest-LoopCoder's loop_num), so one adapter would correct every pass of its layer; Ster steers it but trains no adapters on it"
            );
        }
        if self.lightning.is_some() && !targets.is_empty() {
            bail!(
                "this model's blocks include lightning-attention mixers (MiniMax-Text-01's and Ling 2.5's linear attention) with no attention projection to adapt on every layer; Ster steers it but trains no adapters on it"
            );
        }
        if self.delta_rule.is_some() && !targets.is_empty() {
            bail!(
                "this model's blocks include gated delta-rule mixers (Qwen3-Next's and Kimi-Linear's linear attention) with no attention projection to adapt on every layer; Ster steers it but trains no adapters on it"
            );
        }
        if self.state_space.is_some() && !targets.is_empty() {
            bail!(
                "this model's blocks are state-space mixers (Mamba) with no attention or feed-forward projection to adapt; Ster steers it but trains no adapters on it"
            );
        }
        if let Some(target) = targets
            .iter()
            .find(|target| self.placement(**target, 0, config).is_none())
        {
            let choices: Vec<&str> = Target::ALL
                .into_iter()
                .filter(|choice| self.placement(*choice, 0, config).is_some())
                .map(Target::name)
                .collect();
            let choices = choices.join(", ");
            let why = if self.query_bottleneck.is_some() && *target == Target::Query {
                "its query passes a bottleneck and a norm (Step3's q_proj and inter_norm) before wq"
            } else if self.global_attention.is_some() && !feed_forward_target(*target) {
                "its full-attention layers' heads differ from its sliding-window layers' in count or width (Gemma 4's global_head_dim, MiMo-V2's swa_num_key_value_heads, Step 3.5's attention_other_setting)"
            } else if self.value_experts.is_some() && *target == Target::Value {
                "its routed layers take their value from value experts (K2-Horizon's v_experts) with no v_proj"
            } else if self.latent.is_some() && !feed_forward_target(*target) {
                "its attention is latent (DeepSeek's low-rank query and key-value)"
            } else if self.experts.is_some() {
                "its feed-forward is a mixture of experts"
            } else {
                "its feed-forward has no gate"
            };
            bail!(
                "this model has no single {} projection to adapt because {why}; choose adapter targets among {choices}",
                target.name()
            );
        }
        Ok(())
    }
}

fn feed_forward_target(target: Target) -> bool {
    matches!(target, Target::Gate | Target::Up | Target::Down)
}

/// Where one projection's weight sits in the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub tensor: String,
    /// `None` when the projection is the whole tensor; otherwise the row
    /// blocks it owns, as `(row in the tensor, row in the projection,
    /// rows)`, rows counted in `[outputs, inputs]` orientation.
    pub blocks: Option<Vec<(usize, usize, usize)>>,
    /// The tensor is stored `[inputs, outputs]` (GPT-2's `Conv1D`).
    pub transposed: bool,
    /// The family's root prefix, which a checkpoint saved from the base
    /// model leaves off every name.
    pub root: &'static str,
}

impl Placement {
    fn whole(tensor: String) -> Self {
        Self {
            tensor,
            blocks: None,
            transposed: false,
            root: "",
        }
    }

    fn blocks(tensor: String, blocks: Vec<(usize, usize, usize)>) -> Self {
        Self {
            blocks: Some(blocks),
            ..Self::whole(tensor)
        }
    }

    /// The tensor's name in a checkpoint saved without the root prefix, when
    /// the family has one.
    pub fn without_root(&self) -> Option<String> {
        if self.root.is_empty() {
            return None;
        }
        self.tensor
            .strip_prefix(self.root)
            .and_then(|rest| rest.strip_prefix('.'))
            .map(str::to_owned)
    }
}

/// How a checkpoint stores the query, key and value projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvLayout {
    /// Three tensors.
    Separate,
    /// One tensor, all query rows, then key, then value (Phi-3).
    Stacked,
    /// One tensor by key-value group: each group's query heads, then its key
    /// head, then its value head (Falcon); with a group per head, each head's
    /// query, key and value in turn (GPT-NeoX, BLOOM).
    Grouped,
    /// The query its own tensor, key and value one tensor whose rows hold
    /// each key-value head's key, then its value (TeleChat2's `key_value`).
    PairedKeyValue,
}

/// The function an elementwise attention gate passes `gate_proj` through:
/// K2-Horizon's `attention_gate_func`, AFMoE's sigmoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateFunction {
    Silu,
    /// Softplus with `β = ln 2`: `log2(1 + 2^x)`.
    Softplus,
    Sigmoid,
    /// Softplus, `ln(1 + e^x)` (Laguna's head gate).
    NaturalSoftplus,
}

/// How a token's position enters the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Positions {
    /// Query and key rotated by position.
    Rotary,
    /// A learned table added to the embedding, read `offset` rows in (OPT
    /// keeps two rows before position zero).
    Learned { offset: usize },
    /// A per-head linear bias on the attention scores (BLOOM, MPT, Falcon's
    /// `alibi`); `inside_scale` adds it before the scores are divided by the
    /// head width, as Falcon does, rather than after.
    Alibi { inside_scale: bool },
    /// Step1's bias: `-slope · sqrt(distance)` on the scores after they are
    /// divided by the head width, with ALiBi's slopes.
    AlibiRoot,
    /// No position signal: the recurrence carries order (Mamba).
    None,
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
    /// The shared expert every token goes through beside the routed ones:
    /// Qwen2-MoE's, scaled by a sigmoid gate, or DeepSeek's, added as is.
    pub shared: Option<SharedExpert>,
    pub layout: ExpertLayout,
    /// Bit `i` set means layer `i` keeps a dense feed-forward (Qwen's
    /// `mlp_only_layers` and `decoder_sparse_step`, DeepSeek's
    /// `first_k_dense_replace` and `moe_layer_freq`).
    pub dense_layers: u128,
    /// How the router's logits become expert scores.
    pub scoring: Scoring,
    /// DeepSeek's group-limited routing: the experts fall into `groups`
    /// equal groups, and a token picks among the experts of its best
    /// `chosen_groups` groups only.
    pub groups: Option<ExpertGroups>,
    /// The tensor holding the bias added to the scores to choose experts,
    /// never to weigh them: DeepSeek-V3's `mlp.gate.e_score_correction_bias`,
    /// ERNIE 4.5's `mlp.moe_statics.e_score_correction_bias`, LFM2-MoE's
    /// `feed_forward.expert_bias`.
    pub selection_bias: Option<&'static str>,
    /// `routed_scaling_factor`, multiplying the routed experts' weights.
    pub routed_scale: Option<f64>,
    /// Clamped SwiGLU experts: GPT-OSS's one `swiglu_limit`, or Step 3.5's
    /// limits per layer.
    pub swiglu_limit: Option<SwigluLimit>,
    /// LongCat-Flash's zero-computation experts (`zero_expert_num`, of
    /// `zero_expert_type` `identity`): router outputs past `count` whose
    /// expert returns its input.
    pub identity_experts: usize,
    /// Cohere2-MoE's `shared_expert_combination_strategy` `average`: the
    /// routed and shared experts' sum is halved.
    pub average_shared: bool,
    /// Llama 4: each chosen expert reads its input already multiplied by
    /// its router weight, and its output joins unweighted.
    pub weight_input: bool,
    /// Nemotron-H's latent experts (`moe_latent_size`): the routed experts
    /// work on `fc1_latent_proj(x)`, this wide, and their sum returns
    /// through `fc2_latent_proj`; the router and shared experts read `x`.
    pub latent: Option<usize>,
}

/// Llama 4's attention temperature: on the layers in `layers`, each query
/// at position `p` is multiplied by
/// `1 + beta · ln(1 + floor((p + shift) / interval))` — Llama 4's
/// `attn_scale`, `floor_scale` and one-based positions on its NoPE layers,
/// Ministral 3's `llama_4_scaling_beta` and
/// `original_max_position_embeddings` with zero-based positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryTemperature {
    pub beta: f64,
    pub interval: usize,
    pub shift: usize,
    pub layers: u128,
}

impl QueryTemperature {
    /// Whether layer `layer`'s queries are tempered.
    pub fn at(&self, layer: usize) -> bool {
        layer < 128 && self.layers & (1u128 << layer) != 0
    }
}

/// How a family clamps its experts' SwiGLU.
#[derive(Debug, Clone, PartialEq)]
pub enum SwigluLimit {
    /// GPT-OSS: `(clamp(up, ±limit) + 1) · g · sigmoid(1.702 g)` with
    /// `g = min(gate, limit)`, on every routed expert.
    GptOss(f64),
    /// Step 3.5: `min(silu(gate), limit) · clamp(up, ±limit)`, the routed
    /// experts' limit on layer `i` `routed[i]` (`swiglu_limits`) and the
    /// shared expert's `shared[i]` (`swiglu_limits_shared`); zero means
    /// unclamped.
    Step { routed: Vec<f64>, shared: Vec<f64> },
}

/// A shared expert's inner width and whether a sigmoid gate scales it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedExpert {
    pub intermediate: usize,
    /// Its module below the layer: Qwen2-MoE's `mlp.shared_expert`,
    /// DeepSeek's and ERNIE's `mlp.shared_experts`, HunYuan's
    /// `mlp.shared_mlp`, Granite 4.0's `shared_mlp`, Nemotron-H's
    /// `mixer.shared_experts`.
    pub module: &'static str,
    /// Scaled by the sigmoid of `mlp.shared_expert_gate` (Qwen2-MoE).
    pub gated: bool,
    pub form: SharedForm,
}

/// The projections a shared expert stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedForm {
    /// `gate_proj`, `up_proj` and `down_proj`.
    GateUpDown,
    /// `up_proj` and `down_proj`, no gate (Nemotron-H).
    UpDown,
    /// Gate and up stacked in `input_linear`, down in `output_linear`
    /// (Granite 4.0).
    Stacked,
}

/// How expert scores come from the router's logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scoring {
    Softmax,
    /// DeepSeek-V3's independent sigmoid per expert.
    Sigmoid,
    /// PhiMoE's SparseMixer at inference: each of the `top_k` rounds takes
    /// the best expert not yet chosen, and weighs it by its softmax among
    /// the experts left whose logit lies within `2 · jitter` of the best,
    /// relative to the larger of its own magnitude and the best logit.
    SparseMixer { jitter: f32 },
}

/// Group-limited routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertGroups {
    pub groups: usize,
    pub chosen_groups: usize,
    /// How a group is ranked: by its best expert (DeepSeek-V2's
    /// `group_limited_greedy`) or by the sum of its best two (DeepSeek-V3's
    /// `noaux_tc`).
    pub rank_by_top_two: bool,
}

/// Where a checkpoint keeps its router and experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLayout {
    /// `block_sparse_moe.gate`, `block_sparse_moe.experts.{e}.w1|w3|w2`.
    Mixtral,
    /// `mlp.gate`, `mlp.experts.{e}.gate_proj|up_proj|down_proj`; a shared
    /// expert is Qwen2-MoE's `mlp.shared_expert` or DeepSeek's
    /// `mlp.shared_experts`.
    Qwen,
    /// `block_sparse_moe.router.layer`, and every expert stacked in
    /// `block_sparse_moe.input_linear` and `output_linear`.
    Granite,
    /// `feed_forward.router`, `feed_forward.experts.{e}.gate_proj|up_proj|down_proj`.
    Jamba,
    /// `mlp.gate.wg`, `mlp.experts.{e}.gate_proj|up_proj|down_proj`.
    HunYuan,
    /// `feed_forward.gate`, `feed_forward.experts.{e}.w1|w3|w2` (LFM2-MoE).
    Lfm2,
    /// `mixer.gate`, `mixer.experts.{e}.up_proj|down_proj`, no gate
    /// projection (Nemotron-H).
    NemotronH,
    /// `moe.gate`, and every expert stacked in `moe.gate_proj` and
    /// `moe.up_proj` (`[experts, width, hidden]`) and `moe.down_proj`
    /// (`[experts, hidden, width]`) (Step3).
    Step3,
    /// `mlp.experts.gate_up_proj` and `down_proj`, with biases.
    GptOss,
    /// `ffn.router.layer`, and every expert stacked in
    /// `ffn.experts.mlp.w1`, `v1` and `w2`, each `[experts · width, hidden]`.
    Dbrx,
    /// `router.proj`, `router.scale` and `router.per_expert_scale`, and
    /// every expert stacked in `experts.gate_up_proj` (`[experts, 2 · width,
    /// hidden]`, gate rows first) and `experts.down_proj` (Gemma 4).
    Gemma4,
    /// `mlp.router.gate`, `mlp.experts.{e}.gate_proj|up_proj|down_proj`
    /// (HY V3).
    HyV3,
    /// `mlp.router.classifier`, `mlp.experts.{e}.gate_proj|up_proj|down_proj`
    /// (LongCat-Flash).
    LongCat,
    /// `self_attn.v_router`, `self_attn.v_experts.{e}`: K2-Horizon's value
    /// experts, each one projection whose SiLU output is the value
    /// (`intermediate` wide), with no gate or down projection.
    Mova,
    /// `feed_forward.router`, every expert stacked inputs-first in
    /// `feed_forward.experts.gate_up_proj` (`[experts, hidden, 2 · width]`,
    /// gate columns first) and `down_proj` (`[experts, width, hidden]`),
    /// and `feed_forward.shared_expert` (Llama 4).
    Llama4,
}

/// Where a family keeps its tensors. `embeddings`, `positions`,
/// `embedding_norm`, `layers` and `final_norm` are full paths below `root`;
/// `lm_head` is from the checkpoint root; every other path is below one layer
/// (`{layers}.{i}`). Some checkpoints are saved from the base model and drop
/// `root` (GPT-2's `wte` rather than `transformer.wte`);
/// [`Names::without_root`] is the same layout for those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Names {
    pub root: &'static str,
    pub embeddings: &'static str,
    /// The learned position table, for a family that has one.
    pub positions: &'static str,
    /// The norm straight after the embedding, for a family that has one
    /// (BLOOM).
    pub embedding_norm: &'static str,
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
    /// A state-space mixer, below a layer.
    pub state_space: &'static str,
    /// The per-head or whole-projection query and key norms, below the
    /// attention block.
    pub query_norm: &'static str,
    pub key_norm: &'static str,
    /// The per-head sink logits, below the attention block (GPT-OSS's
    /// `sinks`).
    pub sinks: &'static str,
    /// The prefix a multimodal checkpoint keeps the whole language model
    /// below (Gemma 3's `language_model`); empty for a text-only checkpoint.
    pub wrapper: &'static str,
}

impl Names {
    /// The same layout in a checkpoint saved without the `root` prefix.
    pub fn without_root(self) -> Self {
        let prefix = self.root;
        let strip = |path: &'static str| -> &'static str {
            path.strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix('.'))
                .unwrap_or(path)
        };
        Self {
            root: "",
            embeddings: strip(self.embeddings),
            positions: strip(self.positions),
            embedding_norm: strip(self.embedding_norm),
            layers: strip(self.layers),
            final_norm: strip(self.final_norm),
            ..self
        }
    }

    /// Llama's names; OLMo 2's post-norm block reuses
    /// `post_attention_layernorm` for the norm over attention's output.
    pub const LLAMA: Self = Self {
        root: "model",
        embeddings: "model.embed_tokens",
        positions: "model.embed_positions",
        embedding_norm: "model.embed_layernorm",
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
        state_space: "mamba",
        query_norm: "q_norm",
        key_norm: "k_norm",
        sinks: "sinks",
        wrapper: "",
    };
    /// Granite 4.0 without experts: its `shared_mlp` is the feed-forward,
    /// gate and up stacked in `input_linear`, down in `output_linear`.
    pub const GRANITE_HYBRID: Self = Self {
        fused_gate_up: "shared_mlp.input_linear",
        down: "shared_mlp.output_linear",
        ..Self::LLAMA
    };
    /// Gemma 2 and 3: `post_attention_layernorm` is over attention's output,
    /// so the norm before the feed-forward is `pre_feedforward_layernorm`.
    pub const GEMMA2: Self = Self {
        feed_forward_norm: "pre_feedforward_layernorm",
        ..Self::LLAMA
    };
    /// MuseGlimmer: Gemma 2's sandwich norms, attention's output gate in
    /// `self_attn.gate_proj`, and a weightless norm after the embedding
    /// that names no tensor.
    pub const MUSE_GLIMMER: Self = Self {
        embedding_norm: "",
        ..Self::GEMMA2
    };
    /// HyperCLOVAX: Llama's names, with `post_norm1` and `post_norm2` over
    /// attention's and the feed-forward's outputs.
    pub const HYPERCLOVAX: Self = Self {
        attention_output_norm: "post_norm1",
        feed_forward_output_norm: "post_norm2",
        ..Self::LLAMA
    };
    /// TeleChat2: everything below `transformer` (`word_embeddings`, `h`,
    /// `ln_f`), `self_attention.query` beside one `key_value` matrix, and
    /// `self_attention.dense` as the output.
    pub const TELECHAT: Self = Self {
        root: "transformer",
        embeddings: "transformer.word_embeddings",
        layers: "transformer.h",
        final_norm: "transformer.ln_f",
        attention: "self_attention",
        query: "query",
        fused_qkv: "key_value",
        output: "self_attention.dense",
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
    /// openPangu's sandwich norms: `post_attention_layernorm` over
    /// attention's output, `pre_mlp_layernorm` before the feed-forward and
    /// `post_mlp_layernorm` over its output.
    pub const PANGU_SANDWICH: Self = Self {
        feed_forward_norm: "pre_mlp_layernorm",
        feed_forward_output_norm: "post_mlp_layernorm",
        ..Self::LLAMA
    };
    /// A.X-K1: `post_mlp_layernorm` over each routed layer's feed-forward
    /// output.
    pub const AXK1: Self = Self {
        feed_forward_output_norm: "post_mlp_layernorm",
        ..Self::LLAMA
    };
    /// Llama 4: the dense feed-forward is `feed_forward.{gate,up,down}_proj`.
    pub const LLAMA4: Self = Self {
        gate: Some("feed_forward.gate_proj"),
        up: "feed_forward.up_proj",
        down: "feed_forward.down_proj",
        ..Self::LLAMA
    };
    /// LongCat-Flash's first half of a stored layer: `input_layernorm.0`,
    /// `self_attn.0`, `post_attention_layernorm.0`, `mlps.0`.
    pub const LONGCAT_FIRST: Self = Self {
        attention: "self_attn.0",
        output: "self_attn.0.o_proj",
        attention_norm: "input_layernorm.0",
        feed_forward_norm: "post_attention_layernorm.0",
        gate: Some("mlps.0.gate_proj"),
        up: "mlps.0.up_proj",
        down: "mlps.0.down_proj",
        ..Self::LLAMA
    };
    /// LongCat-Flash's second half of a stored layer, the same names under
    /// `.1`.
    pub const LONGCAT_SECOND: Self = Self {
        attention: "self_attn.1",
        output: "self_attn.1.o_proj",
        attention_norm: "input_layernorm.1",
        feed_forward_norm: "post_attention_layernorm.1",
        gate: Some("mlps.1.gate_proj"),
        up: "mlps.1.up_proj",
        down: "mlps.1.down_proj",
        ..Self::LLAMA
    };
    /// Nanbeige: per-head `q_layernorm` and `k_layernorm`.
    pub const NANBEIGE: Self = Self {
        query_norm: "q_layernorm",
        key_norm: "k_layernorm",
        ..Self::LLAMA
    };
    /// Apertus: `attention_layernorm` before attention and
    /// `feedforward_layernorm` before its plain up-activation-down
    /// feed-forward.
    pub const APERTUS: Self = Self {
        attention_norm: "attention_layernorm",
        feed_forward_norm: "feedforward_layernorm",
        ..Self::UP_DOWN
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
        root: "gpt_neox",
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
        root: "transformer",
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
    /// GPT-2 and GPT-BigCode: `wte` and the learned `wpe`, `h.{i}`, `ln_1`
    /// and `ln_2`, one `c_attn` holding query, key and value, `c_proj`,
    /// `mlp.c_fc`/`mlp.c_proj`, and `ln_f`.
    pub const GPT2: Self = Self {
        root: "transformer",
        embeddings: "transformer.wte",
        positions: "transformer.wpe",
        layers: "transformer.h",
        final_norm: "transformer.ln_f",
        attention: "attn",
        fused_qkv: "c_attn",
        output: "attn.c_proj",
        gate: None,
        up: "mlp.c_fc",
        down: "mlp.c_proj",
        attention_norm: "ln_1",
        feed_forward_norm: "ln_2",
        ..Self::LLAMA
    };
    /// OPT: everything below `model.decoder`, a learned `embed_positions`,
    /// `self_attn_layer_norm` and `final_layer_norm` in each layer,
    /// `out_proj`, `fc1`/`fc2`, and a `final_layer_norm` after the stack.
    pub const OPT: Self = Self {
        embeddings: "model.decoder.embed_tokens",
        positions: "model.decoder.embed_positions",
        layers: "model.decoder.layers",
        final_norm: "model.decoder.final_layer_norm",
        output: "self_attn.out_proj",
        gate: None,
        up: "fc1",
        down: "fc2",
        attention_norm: "self_attn_layer_norm",
        feed_forward_norm: "final_layer_norm",
        ..Self::LLAMA
    };
    /// BLOOM: `word_embeddings` and its `word_embeddings_layernorm`, `h.{i}`,
    /// a head-interleaved `self_attention.query_key_value`, `dense`,
    /// `dense_h_to_4h`/`dense_4h_to_h`, and `ln_f`.
    pub const BLOOM: Self = Self {
        root: "transformer",
        embeddings: "transformer.word_embeddings",
        embedding_norm: "transformer.word_embeddings_layernorm",
        layers: "transformer.h",
        final_norm: "transformer.ln_f",
        attention: "self_attention",
        fused_qkv: "query_key_value",
        output: "self_attention.dense",
        gate: None,
        up: "mlp.dense_h_to_4h",
        down: "mlp.dense_4h_to_h",
        ..Self::LLAMA
    };
    /// Falcon: BLOOM's tensor names without the embedding norm; the parallel
    /// block's one norm is `input_layernorm`.
    pub const FALCON: Self = Self {
        embedding_norm: "transformer.embedding_norm",
        ..Self::BLOOM
    };
    /// Step3: the query projection is `wq`, after the bottleneck `q_proj`
    /// and `inter_norm`; the routed feed-forward is `moe`, the shared one
    /// `share_expert`.
    pub const STEP3: Self = Self {
        query: "wq",
        ..Self::LLAMA
    };
    /// Falcon's new decoder architecture: a parallel block with `ln_attn`
    /// before attention and `ln_mlp` before the feed-forward.
    pub const FALCON_TWO_NORMS: Self = Self {
        attention_norm: "ln_attn",
        feed_forward_norm: "ln_mlp",
        ..Self::FALCON
    };
    /// MPT: `wte` (and a learned `wpe` without ALiBi), `blocks.{i}` with
    /// `norm_1` and `norm_2`, one `attn.Wqkv`, `out_proj`,
    /// `ffn.up_proj`/`ffn.down_proj`, and `norm_f`.
    pub const MPT: Self = Self {
        root: "transformer",
        embeddings: "transformer.wte",
        positions: "transformer.wpe",
        layers: "transformer.blocks",
        final_norm: "transformer.norm_f",
        attention: "attn",
        fused_qkv: "Wqkv",
        output: "attn.out_proj",
        gate: None,
        up: "ffn.up_proj",
        down: "ffn.down_proj",
        attention_norm: "norm_1",
        feed_forward_norm: "norm_2",
        ..Self::LLAMA
    };
    /// Mamba and Falcon-Mamba: everything below `backbone`, `embeddings`,
    /// `layers.{i}` with one `norm` and its `mixer`, and `norm_f`.
    pub const MAMBA: Self = Self {
        root: "backbone",
        embeddings: "backbone.embeddings",
        layers: "backbone.layers",
        final_norm: "backbone.norm_f",
        state_space: "mixer",
        attention_norm: "norm",
        ..Self::LLAMA
    };
    /// Nemotron-H: Mamba's layout, every layer's one sublayer its `mixer` —
    /// a Mamba-2 scan, attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
    /// or a feed-forward (`up_proj`, `down_proj`).
    pub const NEMOTRON_H: Self = Self {
        attention: "mixer",
        output: "mixer.o_proj",
        gate: None,
        up: "mixer.up_proj",
        down: "mixer.down_proj",
        ..Self::MAMBA
    };
    /// Nemotron Puzzle: Nemotron-H's layout below `model` rather than
    /// `backbone`.
    pub const NEMOTRON_H_MODEL: Self = Self {
        root: "model",
        embeddings: "model.embeddings",
        layers: "model.layers",
        final_norm: "model.norm_f",
        ..Self::NEMOTRON_H
    };
    /// Laguna: Llama's names, with the sliding-window layers' sink logits in
    /// `self_attn.sink`.
    pub const LAGUNA: Self = Self {
        sinks: "sink",
        ..Self::LLAMA
    };
    /// ChatGLM (`chatglm`): everything below `transformer`,
    /// `embedding.word_embeddings`, `encoder.layers.{i}` with
    /// `self_attention.query_key_value` and `dense`, `mlp.dense_h_to_4h`
    /// (gate rows above up rows) and `dense_4h_to_h`,
    /// `encoder.final_layernorm`, and `output_layer` as the head.
    pub const CHATGLM: Self = Self {
        root: "transformer",
        embeddings: "transformer.embedding.word_embeddings",
        layers: "transformer.encoder.layers",
        final_norm: "transformer.encoder.final_layernorm",
        lm_head: "transformer.output_layer",
        attention: "self_attention",
        fused_qkv: "query_key_value",
        output: "self_attention.dense",
        fused_gate_up: "mlp.dense_h_to_4h",
        down: "mlp.dense_4h_to_h",
        ..Self::LLAMA
    };
    /// Jamba: `mamba` or `self_attn` after `input_layernorm`,
    /// `pre_ff_layernorm` before `feed_forward`, and `final_layernorm`.
    pub const JAMBA: Self = Self {
        final_norm: "model.final_layernorm",
        gate: Some("feed_forward.gate_proj"),
        up: "feed_forward.up_proj",
        down: "feed_forward.down_proj",
        feed_forward_norm: "pre_ff_layernorm",
        ..Self::LLAMA
    };
    /// Ling (Bailing-MoE): `word_embeddings`, `attention.query_key_value`
    /// with per-head `query_layernorm` and `key_layernorm`, and
    /// `attention.dense` as the output.
    pub const BAILING: Self = Self {
        embeddings: "model.word_embeddings",
        attention: "attention",
        fused_qkv: "query_key_value",
        output: "attention.dense",
        query_norm: "query_layernorm",
        key_norm: "key_layernorm",
        ..Self::LLAMA
    };
    /// InternLM2: `tok_embeddings`, `attention.wqkv` grouped by key-value
    /// head, `attention.wo`, `feed_forward.w1`/`w3`/`w2`, `attention_norm`
    /// and `ffn_norm`, and `output` as the head.
    pub const INTERNLM2: Self = Self {
        embeddings: "model.tok_embeddings",
        lm_head: "output",
        attention: "attention",
        fused_qkv: "wqkv",
        output: "attention.wo",
        gate: Some("feed_forward.w1"),
        up: "feed_forward.w3",
        down: "feed_forward.w2",
        attention_norm: "attention_norm",
        feed_forward_norm: "ffn_norm",
        ..Self::LLAMA
    };
    /// EXAONE 3: everything below `transformer`, `h.{i}` with `ln_1` and
    /// `ln_2`, `attn.attention.{q,k,v,out}_proj`, `mlp.c_fc_0` (gate),
    /// `mlp.c_fc_1` (up) and `mlp.c_proj`, and `ln_f`.
    pub const EXAONE: Self = Self {
        root: "transformer",
        embeddings: "transformer.wte",
        layers: "transformer.h",
        final_norm: "transformer.ln_f",
        attention: "attn.attention",
        output: "attn.attention.out_proj",
        gate: Some("mlp.c_fc_0"),
        up: "mlp.c_fc_1",
        down: "mlp.c_proj",
        attention_norm: "ln_1",
        feed_forward_norm: "ln_2",
        ..Self::LLAMA
    };
    /// HunYuan's per-head query and key norms are `query_layernorm` and
    /// `key_layernorm`.
    pub const HUNYUAN: Self = Self {
        query_norm: "query_layernorm",
        key_norm: "key_layernorm",
        ..Self::LLAMA
    };
    /// MiMo-V2: Llama's names, with the sink logits in
    /// `self_attn.attention_sink_bias`.
    pub const MIMO_V2: Self = Self {
        sinks: "attention_sink_bias",
        ..Self::LLAMA
    };
    /// DBRX: everything below `transformer`, `blocks.{i}` with
    /// `norm_attn_norm.norm_1`, `norm_attn_norm.attn.Wqkv` and `out_proj`,
    /// `norm_attn_norm.norm_2`, and `norm_f`.
    pub const DBRX: Self = Self {
        root: "transformer",
        embeddings: "transformer.wte",
        layers: "transformer.blocks",
        final_norm: "transformer.norm_f",
        attention: "norm_attn_norm.attn",
        fused_qkv: "Wqkv",
        output: "norm_attn_norm.attn.out_proj",
        attention_norm: "norm_attn_norm.norm_1",
        feed_forward_norm: "norm_attn_norm.norm_2",
        ..Self::LLAMA
    };
    /// LFM2: `operator_norm` before the convolution or attention (whose
    /// output projection is `out_proj` and whose head norms are
    /// `q_layernorm` and `k_layernorm`), `ffn_norm` before
    /// `feed_forward.w1`/`w3`/`w2`, and `embedding_norm` after the last block.
    pub const LFM2: Self = Self {
        final_norm: "model.embedding_norm",
        output: "self_attn.out_proj",
        gate: Some("feed_forward.w1"),
        up: "feed_forward.w3",
        down: "feed_forward.w2",
        attention_norm: "operator_norm",
        feed_forward_norm: "ffn_norm",
        query_norm: "q_layernorm",
        key_norm: "k_layernorm",
        state_space: "conv",
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
    /// `long` once it goes beyond; the short table is multiplied by
    /// `short_attention`, the long one by `long_attention` (equal for Phi-3,
    /// PhiMoE's `short_mscale` and `long_mscale` otherwise).
    LongRope {
        short: Vec<f32>,
        long: Vec<f32>,
        original: usize,
        short_attention: f32,
        long_attention: f32,
    },
    /// YaRN: frequencies whose wavelength fits `original` positions more
    /// than `beta_fast` times keep their value, those fitting fewer than
    /// `beta_slow` times are divided by `factor`, a linear ramp blends the
    /// band between, and both tables are multiplied by `attention`.
    Yarn {
        factor: f32,
        original: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention: f32,
    },
    /// Gemma 4's proportional rotation: the first `rotated` frequency pairs
    /// keep their value and the rest are zero, so those components pass
    /// through unrotated; every frequency is divided by `factor`.
    Proportional { rotated: usize, factor: f32 },
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
    /// Llama 4's `use_qk_norm`: a weightless RMS norm over each head's
    /// query and key, after the rotation, on the rotating layers only.
    Unscaled,
    /// MuseGlimmer's QK-norm: a weightless RMS norm over each head's query
    /// and key on every layer, rotating or not.
    Weightless,
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
    /// Plain ReLU (OPT).
    Relu,
    /// xIELU (Apertus), whose parameters are each feed-forward's own
    /// (`mlp.act_fn`); the feed-forward that holds them applies it.
    Xielu,
}

impl Activation {
    pub(crate) fn apply(self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Silu => candle_nn::ops::silu(input),
            Self::GeluTanh => input.gelu(),
            Self::Gelu => input.gelu_erf(),
            Self::Relu2 => input.relu()?.sqr(),
            Self::Relu => input.relu(),
            Self::Xielu => candle_core::bail!(
                "xIELU takes its parameters from the feed-forward that holds them (mlp.act_fn), and this one holds none"
            ),
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
