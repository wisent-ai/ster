//! Where each family keeps its tensors: Llama's layout and the families that
//! override a few of its names, from Granite 4.0 to GLM-4-0414.

mod families;
mod hybrids;

/// How a checkpoint's tensor names relate to the Transformers names Ster
/// reads by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredNames {
    /// Transformers' own names (or a layout the loader recognises from its
    /// files: Mistral's consolidated weights, DeepSeek-V4's native names).
    Transformers,
    /// HY-V4's release names, which Transformers renames on load
    /// (`conversion_mapping.py`, `hy_v4`): `hc_pre` around each
    /// hyper-connection, `hc_attn_layer` and `hc_mlp_layer` for `attn_hc`
    /// and `ffn_hc`, `hc_head_*` for the head's `hc_*`,
    /// `learnable_sink_param` for `sinks` and `linear_gate` for `gate_proj`.
    HyV4,
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
    /// The elementwise attention gate's projection, below the attention
    /// block, when it is a projection of its own.
    pub attention_gate: &'static str,
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
        // A path outside `root` (a top-level head) is kept as it is.
        let strip = |path: &'static str| -> &'static str {
            match path
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix('.'))
            {
                Some(rest) => rest,
                None => path,
            }
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
        attention_gate: "gate_proj",
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
    /// HRM-Text: two stacks of layers with `attn.gqkv_proj` (gate, query,
    /// key and value rows) and `attn.o_proj`, `mlp.gate_up_proj` and
    /// `mlp.down_proj`; the low stack's layers below `model.L_module.layers`,
    /// the high stack's below `model.H_module.layers`.
    pub const HRM: Self = Self {
        attention: "attn",
        fused_qkv: "gqkv_proj",
        output: "attn.o_proj",
        ..Self::LLAMA
    };
    pub const HRM_LOW_LAYERS: &'static str = "model.L_module.layers";
    pub const HRM_HIGH_LAYERS: &'static str = "model.H_module.layers";
    /// Where Transformers saves Kimi-K2.5's decoder layers.
    pub const BLOCKS_LAYERS: &'static str = "model.blocks";
    /// Inkling, as Thinking Machines stores it below `model.llm`: `embed`
    /// and its `embed_norm`, `layers.{i}` with `attn_norm` and `mlp_norm`,
    /// the final `norm`, and the head `unembed`.
    pub const INKLING: Self = Self {
        root: "",
        embeddings: "embed",
        embedding_norm: "embed_norm",
        layers: "layers",
        final_norm: "norm",
        lm_head: "unembed",
        attention_norm: "attn_norm",
        feed_forward_norm: "mlp_norm",
        ..Self::LLAMA
    };
    /// PLaMo 3: layers below `model.layers.layers`, each a `mixer` with one
    /// `qkv_proj` and `o_proj`, a `mlp` with one `gate_up_proj`, and
    /// `pre_mixer_norm`, `post_mixer_norm`, `pre_mlp_norm`, `post_mlp_norm`.
    pub const PLAMO3: Self = Self {
        layers: "model.layers.layers",
        attention: "mixer",
        output: "mixer.o_proj",
        attention_norm: "pre_mixer_norm",
        attention_output_norm: "post_mixer_norm",
        feed_forward_norm: "pre_mlp_norm",
        feed_forward_output_norm: "post_mlp_norm",
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
}
