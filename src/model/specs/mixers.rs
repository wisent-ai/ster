//! The mixers some families run in place of attention — short convolutions,
//! state-space scans, delta rules, lightning attention, shared blocks — and
//! the residual scalings that come with them.

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
    /// Kimi-K3's `use_full_rank_gate`: Kimi's output gate from one full-rank
    /// `g_proj` in place of `g_a_proj` then `g_b_proj`.
    pub full_rank_gate: bool,
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
    Bailing {
        groups: usize,
        silu: bool,
        qk_norm: bool,
    },
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
