//! A routed feed-forward and where a checkpoint keeps it, and Llama 4's
//! query temperature.

use crate::model::Activation;

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
    /// Latent experts: the routed experts work on a narrower projection of
    /// `x`, and their sum returns through a projection back; the router and
    /// shared experts read `x`.
    pub latent: Option<LatentExperts>,
    /// DeepSeek-V4's hash layers (`mlp_layer_types` `hash_moe`): a token's
    /// experts are its row of `{module}.gate.tid2eid`, fixed by its id; the
    /// router still weighs them.
    pub hash_layers: u128,
    /// The activation the routed experts run through when it is not the
    /// model's `hidden_act`: HY-V4's routed experts are SwiGLU whatever its
    /// dense and shared feed-forwards use (`HYV4Experts._apply_gate`).
    pub routed_activation: Option<Activation>,
}

/// Where a mixture's latent experts read and return: Nemotron-H's
/// `moe_latent_size` through `mixer.fc1_latent_proj` and
/// `mixer.fc2_latent_proj`; Kimi-K3's `routed_expert_hidden_size` through
/// `block_sparse_moe.routed_expert_down_proj`, under `latent_moe_use_norm`
/// `routed_expert_norm`, and `routed_expert_up_proj`. Paths are below the
/// layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentExperts {
    pub width: usize,
    pub down: &'static str,
    pub up: &'static str,
    /// An RMS norm over the experts' summed output before `up`, and its
    /// epsilon.
    pub norm: Option<(&'static str, f64)>,
}

impl LatentExperts {
    pub const NEMOTRON_H_DOWN: &'static str = "mixer.fc1_latent_proj";
    pub const NEMOTRON_H_UP: &'static str = "mixer.fc2_latent_proj";
    pub const KIMI_DOWN: &'static str = "block_sparse_moe.routed_expert_down_proj";
    pub const KIMI_UP: &'static str = "block_sparse_moe.routed_expert_up_proj";
    pub const KIMI_NORM: &'static str = "block_sparse_moe.routed_expert_norm";
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
        layer < u128::BITS as usize && self.layers & (1u128 << layer) != 0
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
    /// DeepSeek-V4: `silu(min(gate, limit)) · clamp(up, ±limit)`, on the
    /// routed and the shared experts.
    Inner(f64),
    /// HY-V4: the same clamp as [`SwigluLimit::Inner`], on the routed
    /// experts only; the shared expert is unclamped (vLLM's `HYV4MoEFused`).
    RoutedInner(f64),
    /// MiniMax-M3's SwiGLU-OAI: GPT-OSS's clamp with the gate's sharpness
    /// stated (`swiglu_alpha`), `(clamp(up, ±limit) + 1) · g · sigmoid(alpha
    /// g)` with `g = min(gate, limit)`, on the routed and the shared experts.
    Oai { limit: f64, alpha: f64 },
}

/// How a family clamps its dense feed-forward's SwiGLU: GLM-5-Next's
/// [`SwigluLimit::Inner`] clamp (`swiglu_limit`) or MiniMax-M3's
/// [`SwigluLimit::Oai`] one (`swiglu_limit`, `swiglu_alpha`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenseClamp {
    Inner(f64),
    Oai { limit: f64, alpha: f64 },
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
    SparseMixer {
        jitter: f32,
    },
    /// DeepSeek-V4's `sqrtsoftplus`: `sqrt(softplus(logit))` per expert.
    SqrtSoftplus,
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
    /// DeepSeek-V4: `mlp.gate`, `mlp.experts.{e}.w1|w3|w2` (gate, up,
    /// down) or stacked `mlp.experts.gate_up_proj` and `down_proj`.
    DeepseekV4,
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
