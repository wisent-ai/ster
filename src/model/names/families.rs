//! The tensor names of Starcoder2 through Falcon's two-norm block: the plain
//! feed-forward families, the GPT lineage, BLOOM and Falcon, and the Llama
//! variants beside them.

use super::Names;

impl Names {
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
}
