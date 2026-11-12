//! The tensor names of MPT through LFM2: the state-space and hybrid families
//! (Mamba, Nemotron-H, Jamba, LFM2) and the fused-projection ones beside
//! them.

use super::Names;

impl Names {
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
