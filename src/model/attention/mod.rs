//! Grouped-query attention with adapters on its projections: the scores, the
//! rotary angles they are taken at, and the masks that hide what a position
//! may not see.

pub(super) mod attend;
mod blocks;
mod forward;
#[path = "positions/indexer.rs"]
mod indexer;
#[path = "positions/latent.rs"]
mod latent;
mod load;
mod mix;
#[path = "positions/relative.rs"]
pub(super) mod relative;
#[path = "positions/rope.rs"]
pub(super) mod rope;

use candle_core::Tensor;
use candle_nn::{Linear, Module};

use crate::lora::Adapter;

use super::{
    GateFunction, Pass, QueryKeyNorm, QueryTemperature, Route,
    layer::{experts::Experts, norm::Norm},
};
use latent::Latent;
/// T5's attention masks its scores the way this one does.
pub(super) use attend::masked_fill;

#[derive(Debug, Clone)]
pub(super) struct Attention {
    projections: Projections,
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
    /// Normalise each head's query and key after the rotation rather than
    /// before it (HunYuan).
    norm_after_rotary: bool,
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    /// Width of one head's value, which latent attention makes narrower than
    /// the query.
    value_dim: usize,
    /// How many keys behind it a query may see, on a sliding-window layer.
    window: Option<usize>,
    rotary: Rotary,
    /// Rotate adjacent pairs (Cohere) instead of halves.
    interleaved: bool,
    score_divisor: f64,
    softcap: Option<f64>,
    /// HRM-Text's PrefixLM mask: a prompt read in one pass from position zero
    /// sees itself both ways.
    prefix_bidirectional: bool,
    /// ALiBi's slope per head, `[1, heads, 1, 1]`, for a family whose
    /// positions are a linear bias on the scores (BLOOM, MPT).
    alibi: Option<Tensor>,
    /// The bias is `-slope · sqrt(distance)` (Step1) rather than linear.
    alibi_root: bool,
    /// GPT-OSS's learned sink per head, `[1, heads, 1, 1]` in F32: one extra
    /// logit every query's softmax divides by and no value is read for.
    sinks: Option<Tensor>,
    /// The bound every query, key and value component is clamped to.
    clip_qkv: Option<f64>,
    /// Falcon-H1's `key_multiplier` on every key.
    key_scale: Option<f64>,
    /// MiMo-V2's `attention_value_scale` on every value.
    value_scale: Option<f64>,
    /// Llama 4's query temperature, on this layer when it applies here.
    query_temperature: Option<QueryTemperature>,
    /// Qwen3-Next's gate rows of `q_proj`: attention's output is multiplied
    /// by their sigmoid before the output projection.
    output_gate: Option<Linear>,
    /// Step 3.5's and Laguna's `g_proj`: one logit per head, each head's
    /// output multiplied by it through the gate function.
    head_gate: Option<(Linear, GateFunction)>,
    /// K2-Horizon's `gate_proj` and the function its output passes through
    /// before it multiplies attention's output.
    elementwise_gate: Option<(Linear, GateFunction)>,
    /// K2-Horizon's value experts, in place of `v_proj` on its routed
    /// layers.
    value_experts: Option<Experts>,
    /// Step3's query bottleneck: `q_proj` down to `share_q_dim` and the RMS
    /// norm `inter_norm`, whose output the query projection (`wq`) reads in
    /// place of the hidden state.
    query_bottleneck: Option<(Linear, Norm)>,
    /// Zamba2's per-invocation low-rank terms on query, key and value, set
    /// by [`Attention::with_low_rank`] on each invocation's copy of the
    /// shared block.
    low_rank: Option<Box<[LowRank; 3]>>,
    /// Gemma 4's scale-free norm over each head's value.
    value_norm: Option<Norm>,
    /// This layer's keys and values are the ones Gemma 4's key-value-sharing
    /// layers of its kind reuse; each call leaves them in the cache's
    /// `shared` slot.
    stores_key_values: bool,
    /// IQuest-LoopCoder's gated mix on its later passes.
    loop_mix: Option<LoopMix>,
    /// A GLM-5 `shared` layer: it reuses the last indexed layer's choice.
    index_shared: bool,
    /// MiniMax Sparse Attention's block indexer on this layer (MiniMax-M3).
    block_indexer: Option<blocks::BlockIndexer>,
}

/// IQuest-LoopCoder's mix: on a layer past the first `physical`, the
/// attention is `g · global + (1 − g) · local`, global over the first pass's
/// keys and values of the same stored layer, local over the pass's own
/// within `window`, `g = sigmoid(weight_h · q_h + bias_h)` per head from
/// `gate_projections.{layer}`, both in F32 shaped to broadcast over
/// `[batch, heads, sequence, head_dim]`.
#[derive(Debug, Clone)]
struct LoopMix {
    physical: usize,
    window: usize,
    weight: Tensor,
    bias: Tensor,
}

/// A low-rank term `up(down(x))` the base model adds to a projection.
#[derive(Debug, Clone)]
pub(super) struct LowRank {
    pub down: Linear,
    pub up: Linear,
}

impl LowRank {
    pub(super) fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.up.forward(&self.down.forward(input)?)
    }
}

/// Which rotation this layer rotates its query and key with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rotary {
    Global,
    /// The sliding-window table (Gemma 3's `rope_local_base_freq`).
    Local,
    /// No rotation at all (SmolLM3's NoPE layers, Cohere 2's global layers).
    None,
}

impl Attention {
    /// This attention with Zamba2's per-invocation low-rank terms on query,
    /// key and value, in that order.
    pub(super) fn with_low_rank(mut self, low_rank: [LowRank; 3]) -> Self {
        self.low_rank = Some(Box::new(low_rank));
        self
    }

    /// The window this layer attends through, if it is a sliding-window layer.
    pub(super) fn window(&self) -> Option<usize> {
        self.window
    }
}

/// Where a layer's query, key and value come from.
#[derive(Debug, Clone)]
enum Projections {
    /// One projection each (possibly row views of one fused tensor); no
    /// value projection where the value is the key projection (Gemma 4's
    /// `attention_k_eq_v`).
    Standard {
        query: Linear,
        key: Linear,
        value: Option<Linear>,
    },
    /// DeepSeek's low-rank bottlenecks.
    Latent(Latent),
    /// A Gemma 4 key-value-sharing layer: its query, and the layer whose
    /// keys and values it reuses.
    QueryOnly { query: Linear, source: usize },
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

/// A gate's logits through `function`, composed of ops that have a backward
/// pass: `silu(g)`; softplus with `β = ln 2`, `log2(1 + 2^g)`, and plain
/// softplus, `ln(1 + e^g)`, written as `max(g, 0) + ln(1 + e^-β|g|) / β` so
/// neither overflows; or the sigmoid.
fn gated(logits: &Tensor, function: GateFunction) -> candle_core::Result<Tensor> {
    let softplus = |slope: f64| -> candle_core::Result<Tensor> {
        let tail = ((logits.abs()?.neg()? * slope)?.exp()? + 1.0)?.log()?;
        logits.relu()? + (tail / slope)?
    };
    match function {
        GateFunction::Silu => candle_nn::ops::silu(logits),
        GateFunction::Softplus => softplus(std::f64::consts::LN_2),
        GateFunction::NaturalSoftplus => softplus(1.0),
        GateFunction::Sigmoid => (logits.neg()?.exp()? + 1.0)?.recip(),
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
