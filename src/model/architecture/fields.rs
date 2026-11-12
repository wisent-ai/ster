//! The switches one [`Architecture`] holds.

use crate::model::{
    Activation, BlockIndexSpec, CompressedSpec, DeltaRuleSpec, FeedForwardKind, GateFunction,
    GlobalAttention, HyperConnections, IndexerSpec, InklingSpec, LatentAttention, LayerPlan,
    LightningSpec, Loops, MixtureOfExperts, Names, NgramSpec, NormKind, ParallelScan,
    PerLayerInputSpec, Positions, QkvLayout, QueryKeyNorm, QueryTemperature, Recurrence,
    RopeScaling, ScaledResiduals, SharedBlocksSpec, ShortConvolution, SkipConnections,
    StateSpaceSpec, StoredNames, WindowLatent, DenseClamp,
};

/// How a checkpoint's decoder differs from the plain Llama block, each switch
/// read from the checkpoint's own `config.json`; the parent module lists them.
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
    /// Gemma 3n's AltUp: this many copies of the hidden state carried
    /// through the stack (`altup_num_inputs`), the blocks running on the
    /// first.
    pub altup_streams: Option<usize>,
    /// Gemma 3n's Laurel rank (`laurel_rank`): a low-rank residual beside
    /// attention, averaged with it.
    pub laurel_rank: Option<usize>,
    /// Gemma 3n's activation sparsity: on the layers in the set, the
    /// feed-forward gate keeps only what lies above its mean plus this many
    /// standard deviations (`activation_sparsity_pattern` read as a normal
    /// quantile).
    pub activation_sparsity: Option<(f64, u128)>,
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
    /// HRM-Text's two recurrent stacks.
    pub recurrence: Option<Recurrence>,
    /// HRM-Text's `prefix_lm`: a prompt read whole from position zero is
    /// one bidirectional block, as Transformers reads it given
    /// `token_type_ids` of ones.
    pub prefix_lm: bool,
    /// Kimi-K3's `attn_res_block_size`: attention residuals over blocks of
    /// this many layers, each layer's `self_attention_res_norm`,
    /// `self_attention_res_proj`, `mlp_res_norm` and `mlp_res_proj`, and the
    /// model's `output_attn_res_norm` and `output_attn_res_proj` before the
    /// final norm.
    pub depth_block_size: Option<usize>,
    /// Inkling's relative-position attention, short convolutions and
    /// feed-forward.
    pub inkling: Option<InklingSpec>,
    /// DeepSeek-V4's compressed attention.
    pub compressed: Option<CompressedSpec>,
    /// DeepSeek-V4's and GLM-5-Next's hyper-connected residual streams.
    pub hyper_connections: Option<HyperConnections>,
    /// The dense feed-forward's clamped SwiGLU (GLM-5-Next, MiniMax-M3).
    pub dense_swiglu_limit: Option<DenseClamp>,
    /// Inkling's `unpadded_vocab_size`: logits past it are never scored.
    pub vocabulary_limit: Option<usize>,
    /// DeciLM's per-layer plan, one entry per layer.
    pub layer_plans: Option<Vec<LayerPlan>>,
    /// DeepSeek Sparse Attention's indexer (DeepSeek-V3.2, GLM-5).
    pub sparse_index: Option<IndexerSpec>,
    /// MiniMax-M3's block indexer (MiniMax Sparse Attention).
    pub block_index: Option<BlockIndexSpec>,
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
    /// What attention's and the feed-forward's output norms add to their
    /// stored scales in place of the block's offset (PLaMo 3's 1/5 and
    /// 5^-1.5).
    pub output_norm_shifts: Option<(f64, f64)>,
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
    /// Dots3-Note's sliding-window layers' own latent attention, in place of
    /// `latent` on the layers that attend through the window.
    pub window_latent: Option<WindowLatent>,
    /// Dots3-Note's `k_rope_only_layernorm`: latent attention norms the
    /// rotated key part it shares across heads before rotating it.
    pub rotated_key_norm: bool,
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
    /// How the checkpoint spells the tensors Ster asks for by Transformers'
    /// names.
    pub stored_names: StoredNames,
}
