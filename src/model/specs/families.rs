//! Inkling's decoder, hyper-connected residual streams and DeepSeek-V4's
//! compressed attention.

/// Inkling's decoder (`inkling_text`): every layer's attention scores keys
/// by a relative-position bias in place of a rotation, its keys and values
/// and both sublayers' outputs pass a residual short convolution, and its
/// feed-forward is dense or routed as `routed_layers` says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InklingSpec {
    /// The full-attention layers' heads, key-value heads, head width and
    /// relative extent (`rel_extent`).
    pub full: RelativeHeads,
    /// The sliding layers' (`swa_*`), their extent the window
    /// (`sliding_window_size`).
    pub sliding: RelativeHeads,
    /// The layers that attend through the window.
    pub sliding_layers: u128,
    /// `d_rel`: how many bias-versus-distance profiles each head mixes.
    pub profiles: usize,
    /// `conv_kernel_size` (`sconv_kernel_size`).
    pub kernel: usize,
    /// `log_scaling_alpha` and `log_scaling_n_floor`: on the full layers,
    /// the scores at position `p` times `1 + alpha · ln(max((p + 1) /
    /// floor, 1))`.
    pub log_scaling: Option<(f64, f64)>,
    /// The layers whose feed-forward is routed (`mlp_layer_types`
    /// `sparse`).
    pub routed_layers: u128,
    /// `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`,
    /// `moe_intermediate_size` and `route_scale`.
    pub experts: usize,
    pub shared: usize,
    pub top_k: usize,
    pub expert_intermediate: usize,
    pub route_scale: f64,
}

/// Hyper-connections (DeepSeek-V4, GLM-5-Next, HY-V4): `streams` residual
/// streams (`hc_mult`), each sublayer reading them collapsed and its output
/// spread back over them, as `form` says (`hc_eps` beside the gates). The
/// model's last streams are collapsed by `hc_head` (DeepSeek-V4, HY-V4) or
/// averaged (GLM-5-Next).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperConnections {
    pub streams: usize,
    pub form: HyperForm,
    pub eps: f64,
    pub learned_head: bool,
}

/// How a hyper-connection spreads a sublayer's output back over the
/// streams.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperForm {
    /// Manifold-constrained (DeepSeek-V4, GLM-5-Next): the incoming streams
    /// are mixed by a matrix `sinkhorn_iterations` Sinkhorn-Knopp rounds make
    /// doubly stochastic (`hc_sinkhorn_iters`), and the output joins them
    /// weighted by twice a sigmoid.
    Manifold { sinkhorn_iterations: usize },
    /// Independent (HY-V4's `enable_ihc`): no mixing matrix, each incoming
    /// stream passes as it is, and the output joins it weighted by
    /// `magnitude` (`hc_magnitude`) times a sigmoid, plus `hc_eps`.
    Independent { magnitude: f64 },
}

/// DeepSeek-V4's attention (`deepseek_v4`): one key-value head that is both
/// key and value, attended through `sliding_window` and, on the compressed
/// layers, beside entries that summarise every `rate` earlier tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressedSpec {
    pub heads: usize,
    pub head_dim: usize,
    /// The trailing slice of each head that rotates (`qk_rope_head_dim`).
    pub rotary_dim: usize,
    /// `q_lora_rank`.
    pub query_rank: usize,
    /// `o_groups` and `o_lora_rank`: the output projection's groups and
    /// each group's width.
    pub output_groups: usize,
    pub output_rank: usize,
    pub window: usize,
    /// The layers attending beside compressed sparse entries (`csa`) and
    /// heavily compressed ones (`hca`), and their rates.
    pub sparse_layers: u128,
    pub heavy_layers: u128,
    pub sparse_rate: usize,
    pub heavy_rate: usize,
    /// The sparse layers' indexer: `index_n_heads`, `index_head_dim`,
    /// `index_topk`.
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_top_k: usize,
    /// The sliding layers' rotation base (`rope_theta`) and the compressed
    /// layers' (`compress_rope_theta`), whose YaRN is `compress_yarn`.
    pub main_theta: f32,
    pub compress_theta: f32,
    pub compress_yarn: Option<CompressYarn>,
}

/// The YaRN DeepSeek-V4's compressed layers rotate by, its magnitude one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressYarn {
    pub factor: f32,
    pub original: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub truncate: bool,
}

/// One kind of Inkling attention layer's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelativeHeads {
    pub heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub extent: usize,
}
