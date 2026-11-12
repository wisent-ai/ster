//! How attention departs from the plain block: projection layouts, gates,
//! positions, the full-attention layers' own shape, per-layer inputs, n-gram
//! embeddings and latent attention.

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
    /// One tensor: the elementwise attention gate's rows, then query, key and
    /// value rows (HRM-Text's `gqkv_proj`).
    GateStacked,
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

/// Dots3-Note's sliding-window attention (`swa_num_attention_heads`,
/// `swa_q_lora_rank`, `swa_kv_lora_rank`, `swa_qk_nope_head_dim`,
/// `swa_qk_rope_head_dim`, `swa_v_head_dim`): `heads` query heads of its own
/// latent attention, scored over its own head width, with its own
/// bottleneck scales (`apply_mla_qkv_lora_rescale` against its own ranks).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowLatent {
    pub heads: usize,
    pub latent: LatentAttention,
    pub scales: Option<(f64, f64)>,
}
