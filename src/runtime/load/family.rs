//! Which decoder family a checkpoint is, read from its `config.json`.
//!
//! Every switch below is a key the checkpoint publishes; none is a value Ster
//! picks. A config that asks for something the decoder does not implement is
//! refused with the key that asked for it, instead of being loaded wrong.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{
    ALIBI_SPAN, Activation, Architecture, DeltaRuleForm, DeltaRuleSpec, ExpertGroups, ExpertLayout, FeedForwardKind,
    GateFunction, GlobalAttention, IndexerSpec, LatentAttention, LayerPlan, LightningForm, LightningSpec, Loops, MixtureOfExperts,
    Names, NgramSpec, NormKind, ParallelScan, ParameterNorm,
    PerLayerInputSpec, Positions, QkvLayout, QueryKeyNorm, QueryTemperature, RopeScaling, ScaledResiduals, Scoring, SharedBlocksSpec,
    SharedExpert, SharedForm, ShortConvolution, SkipConnections, StateSpaceSpec, StructuredSpec, SwigluLimit,
};

/// Mamba's `time_step_rank: "auto"` is the model width over this, rounded up,
/// as Transformers' `MambaConfig` computes it.
const MAMBA_WIDTH_PER_STEP_RANK: usize = 16;

/// A decoder layout Ster implements, one per Transformers `model_type`.
///
/// The config's `model_type` field names the layout; [`Family::of`] reads it
/// and [`Family::ALL`] is what the refusal of any other value lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Family {
    Llama,
    Mistral,
    Mixtral,
    Qwen2,
    Qwen2Moe,
    Qwen3,
    Qwen3Moe,
    Phi,
    Phi3,
    Granite,
    GraniteMoe,
    StableLm,
    Starcoder2,
    Cohere,
    Cohere2,
    Nemotron,
    Olmo,
    Olmo2,
    Olmo3,
    Olmoe,
    Exaone4,
    InternLm3,
    SeedOss,
    Arcee,
    Ernie45,
    MiniCpm,
    Orion,
    Glm,
    Glm4,
    GptNeox,
    GptJ,
    SmolLm3,
    Gpt2,
    GptBigCode,
    Opt,
    Bloom,
    Falcon,
    Mpt,
    DeepseekV2,
    DeepseekV3,
    MiniCpm3,
    Mamba,
    FalconMamba,
    Glm4Moe,
    InternLm2,
    Exaone,
    Jamba,
    HunYuanDense,
    Mamba2,
    Bamba,
    GptOss,
    Lfm2,
    Ernie45Moe,
    Dbrx,
    Phimoe,
    HunYuanMoe,
    Telechat,
    Lfm2Moe,
    GraniteMoeHybrid,
    NemotronH,
    Jais2,
    BailingMoe,
    TeleFlm,
    FalconH1,
    Qwen3Next,
    KimiLinear,
    MinimaxM2,
    Step3Text,
    MinimaxText,
    Minimax,
    Zamba2,
    Gemma,
    Gemma2,
    Gemma3Text,
    Gemma4Text,
    Gemma4UnifiedText,
    Mimo,
    Mellum,
    FlexOlmo,
    GraniteSwa,
    GraniteMoeSwa,
    GraniteMoeShared,
    Glm4MoeLite,
    IQuestCoder,
    HyperClovaX,
    VaultGemma,
    Apertus,
    ExaoneMoe,
    Solar,
    Step1,
    TeleChat3,
    Param2Moe,
    PanguEmbedded,
    OlmoHybrid,
    HyV3,
    Nanbeige,
    IQuestLoopCoder,
    NemotronNas,
    DeepseekV32,
    GlmMoeDsa,
    Axk1,
    BailingHybrid,
    MimoV2Flash,
    MimoV2,
    Step3p5,
    K2Horizon,
    PanguUltraMoe,
    LongcatFlash,
    SarvamMoe,
    SarvamMla,
    Cohere2Moe,
    DeepseekMoe,
    Ministral3,
    Llama4Text,
    Qwen35Text,
    Qwen35MoeText,
    Afmoe,
    NemotronHPuzzle,
    ChatGlm,
    Laguna,
    LongcatFlashNgram,
    MuseGlimmerText,
}

impl Family {
    pub(super) const ALL: [Self; 122] = [
        Self::Llama,
        Self::Mistral,
        Self::Mixtral,
        Self::Qwen2,
        Self::Qwen2Moe,
        Self::Qwen3,
        Self::Qwen3Moe,
        Self::Phi,
        Self::Phi3,
        Self::Granite,
        Self::GraniteMoe,
        Self::StableLm,
        Self::Starcoder2,
        Self::Cohere,
        Self::Cohere2,
        Self::Nemotron,
        Self::Olmo,
        Self::Olmo2,
        Self::Olmo3,
        Self::Olmoe,
        Self::Exaone4,
        Self::InternLm3,
        Self::SeedOss,
        Self::Arcee,
        Self::Ernie45,
        Self::MiniCpm,
        Self::Orion,
        Self::Glm,
        Self::Glm4,
        Self::GptNeox,
        Self::GptJ,
        Self::SmolLm3,
        Self::Gpt2,
        Self::GptBigCode,
        Self::Opt,
        Self::Bloom,
        Self::Falcon,
        Self::Mpt,
        Self::DeepseekV2,
        Self::DeepseekV3,
        Self::MiniCpm3,
        Self::Mamba,
        Self::FalconMamba,
        Self::Glm4Moe,
        Self::InternLm2,
        Self::Exaone,
        Self::Jamba,
        Self::HunYuanDense,
        Self::Mamba2,
        Self::Bamba,
        Self::GptOss,
        Self::Lfm2,
        Self::Ernie45Moe,
        Self::Dbrx,
        Self::Phimoe,
        Self::HunYuanMoe,
        Self::Telechat,
        Self::Lfm2Moe,
        Self::GraniteMoeHybrid,
        Self::NemotronH,
        Self::Jais2,
        Self::BailingMoe,
        Self::TeleFlm,
        Self::FalconH1,
        Self::Qwen3Next,
        Self::KimiLinear,
        Self::MinimaxM2,
        Self::Step3Text,
        Self::MinimaxText,
        Self::Minimax,
        Self::Zamba2,
        Self::Gemma,
        Self::Gemma2,
        Self::Gemma3Text,
        Self::Gemma4Text,
        Self::Gemma4UnifiedText,
        Self::Mimo,
        Self::Mellum,
        Self::FlexOlmo,
        Self::GraniteSwa,
        Self::GraniteMoeSwa,
        Self::GraniteMoeShared,
        Self::Glm4MoeLite,
        Self::IQuestCoder,
        Self::HyperClovaX,
        Self::VaultGemma,
        Self::Apertus,
        Self::ExaoneMoe,
        Self::Solar,
        Self::Step1,
        Self::TeleChat3,
        Self::Param2Moe,
        Self::PanguEmbedded,
        Self::OlmoHybrid,
        Self::HyV3,
        Self::Nanbeige,
        Self::IQuestLoopCoder,
        Self::NemotronNas,
        Self::DeepseekV32,
        Self::GlmMoeDsa,
        Self::Axk1,
        Self::BailingHybrid,
        Self::MimoV2Flash,
        Self::MimoV2,
        Self::Step3p5,
        Self::K2Horizon,
        Self::PanguUltraMoe,
        Self::LongcatFlash,
        Self::SarvamMoe,
        Self::SarvamMla,
        Self::Cohere2Moe,
        Self::DeepseekMoe,
        Self::Ministral3,
        Self::Llama4Text,
        Self::Qwen35Text,
        Self::Qwen35MoeText,
        Self::Afmoe,
        Self::NemotronHPuzzle,
        Self::ChatGlm,
        Self::Laguna,
        Self::LongcatFlashNgram,
        Self::MuseGlimmerText,
    ];

    /// The family a config's `model_type` names, if Ster implements it.
    pub(super) fn of(model_type: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|family| family.model_type() == model_type)
    }

    /// The family a config without `model_type` belongs to, from the model
    /// class its `architectures` names, for the families whose published
    /// configs leave `model_type` to their remote-code config class.
    pub(super) fn of_architectures(raw: &Value) -> Option<Self> {
        let names = raw.get("architectures")?.as_array()?;
        Self::ALL.into_iter().find(|family| {
            family.remote_class().is_some_and(|class| names.iter().any(|name| name.as_str() == Some(class)))
        })
    }

    /// The remote-code model class that stands in for `model_type` in this
    /// family's configs, where they omit it.
    fn remote_class(self) -> Option<&'static str> {
        match self {
            Self::LongcatFlashNgram => Some("LongcatFlashNgramForCausalLM"),
            _ => None,
        }
    }

    /// The `model_type` value Transformers writes for this family.
    pub(super) fn model_type(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::Qwen2 => "qwen2",
            Self::Qwen2Moe => "qwen2_moe",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Phi => "phi",
            Self::Phi3 => "phi3",
            Self::Granite => "granite",
            Self::GraniteMoe => "granitemoe",
            Self::StableLm => "stablelm",
            Self::Starcoder2 => "starcoder2",
            Self::Cohere => "cohere",
            Self::Cohere2 => "cohere2",
            Self::Nemotron => "nemotron",
            Self::Olmo => "olmo",
            Self::Olmo2 => "olmo2",
            Self::Olmo3 => "olmo3",
            Self::Olmoe => "olmoe",
            Self::Exaone4 => "exaone4",
            Self::InternLm3 => "internlm3",
            Self::SeedOss => "seed_oss",
            Self::Arcee => "arcee",
            Self::Ernie45 => "ernie4_5",
            Self::MiniCpm => "minicpm",
            Self::Orion => "orion",
            Self::Glm => "glm",
            Self::Glm4 => "glm4",
            Self::GptNeox => "gpt_neox",
            Self::GptJ => "gptj",
            Self::SmolLm3 => "smollm3",
            Self::Gpt2 => "gpt2",
            Self::GptBigCode => "gpt_bigcode",
            Self::Opt => "opt",
            Self::Bloom => "bloom",
            Self::Falcon => "falcon",
            Self::Mpt => "mpt",
            Self::DeepseekV2 => "deepseek_v2",
            Self::DeepseekV3 => "deepseek_v3",
            Self::MiniCpm3 => "minicpm3",
            Self::Mamba => "mamba",
            Self::FalconMamba => "falcon_mamba",
            Self::Glm4Moe => "glm4_moe",
            Self::InternLm2 => "internlm2",
            Self::Exaone => "exaone",
            Self::Jamba => "jamba",
            Self::HunYuanDense => "hunyuan_v1_dense",
            Self::Mamba2 => "mamba2",
            Self::Bamba => "bamba",
            Self::GptOss => "gpt_oss",
            Self::Lfm2 => "lfm2",
            Self::Ernie45Moe => "ernie4_5_moe",
            Self::Dbrx => "dbrx",
            Self::Phimoe => "phimoe",
            Self::HunYuanMoe => "hunyuan_v1_moe",
            Self::Telechat => "telechat",
            Self::Lfm2Moe => "lfm2_moe",
            Self::GraniteMoeHybrid => "granitemoehybrid",
            Self::NemotronH => "nemotron_h",
            Self::Jais2 => "jais2",
            Self::BailingMoe => "bailing_moe",
            Self::TeleFlm => "TeleFLM",
            Self::FalconH1 => "falcon_h1",
            Self::Qwen3Next => "qwen3_next",
            Self::KimiLinear => "kimi_linear",
            Self::MinimaxM2 => "minimax_m2",
            Self::Step3Text => "step3_text",
            Self::MinimaxText => "minimax_text_01",
            Self::Minimax => "minimax",
            Self::Zamba2 => "zamba2",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Gemma3Text => "gemma3_text",
            Self::Gemma4Text => "gemma4_text",
            Self::Gemma4UnifiedText => "gemma4_unified_text",
            Self::Mimo => "mimo",
            Self::Mellum => "mellum",
            Self::FlexOlmo => "flex_olmo",
            Self::GraniteSwa => "granite_swa",
            Self::GraniteMoeSwa => "granitemoe_swa",
            Self::GraniteMoeShared => "granitemoeshared",
            Self::Glm4MoeLite => "glm4_moe_lite",
            Self::IQuestCoder => "iquestcoder",
            Self::HyperClovaX => "hyperclovax",
            Self::VaultGemma => "vaultgemma",
            Self::Apertus => "apertus",
            Self::ExaoneMoe => "exaone_moe",
            Self::Solar => "solar",
            Self::Step1 => "step1",
            Self::TeleChat3 => "telechat3",
            Self::Param2Moe => "param2moe",
            Self::PanguEmbedded => "PanguEmbedded",
            Self::OlmoHybrid => "olmo_hybrid",
            Self::HyV3 => "hy_v3",
            Self::Nanbeige => "nanbeige",
            Self::IQuestLoopCoder => "iquestloopcoder",
            Self::NemotronNas => "nemotron-nas",
            Self::DeepseekV32 => "deepseek_v32",
            Self::GlmMoeDsa => "glm_moe_dsa",
            Self::Axk1 => "axk1",
            Self::BailingHybrid => "bailing_hybrid",
            Self::MimoV2Flash => "mimo_v2_flash",
            Self::MimoV2 => "mimo_v2",
            Self::Step3p5 => "step3p5",
            Self::K2Horizon => "k2_horizon",
            Self::PanguUltraMoe => "pangu_ultra_moe",
            Self::LongcatFlash => "longcat_flash",
            Self::SarvamMoe => "sarvam_moe",
            Self::SarvamMla => "sarvam_mla",
            Self::Cohere2Moe => "cohere2_moe",
            Self::DeepseekMoe => "deepseek",
            Self::Ministral3 => "ministral3",
            Self::Llama4Text => "llama4_text",
            Self::Qwen35Text => "qwen3_5_text",
            Self::Qwen35MoeText => "qwen3_5_moe_text",
            Self::Afmoe => "afmoe",
            Self::NemotronHPuzzle => "nemotron_h_puzzle",
            Self::ChatGlm => "chatglm",
            Self::Laguna => "laguna",
            Self::LongcatFlashNgram => "longcat_flash_ngram",
            Self::MuseGlimmerText => "muse_glimmer_text",
        }
    }

    /// Whether the family's Transformers config class leaves
    /// `tie_word_embeddings` at the library default, true, so its configs
    /// often omit the key.
    pub(super) fn tied_by_default(self) -> bool {
        matches!(
            self,
            Self::Gemma
                | Self::Gemma2
                | Self::Gemma3Text
                | Self::Gemma4Text
                | Self::Gemma4UnifiedText
                | Self::VaultGemma
                | Self::Cohere
                | Self::Cohere2
                | Self::Cohere2Moe
                | Self::Starcoder2
                | Self::Ernie45
                | Self::Ernie45Moe
                | Self::Gpt2
                | Self::GptBigCode
                | Self::Opt
                | Self::Bloom
                | Self::Falcon
                | Self::Mpt
                | Self::Mamba
                | Self::FalconMamba
        )
    }
}

/// OPT's learned position table keeps this many rows before position zero;
/// Transformers' `OPTLearnedPositionalEmbedding` reads `position + offset`.
const OPT_POSITION_OFFSET: usize = 2;

/// Transformers' `sparsemixer` (`models/phimoe/modeling_phimoe.py`) runs
/// exactly two selection rounds, so PhiMoE routes two experts per token.
const SPARSE_MIXER_EXPERTS: usize = 2;

/// OLMo 1's and DBRX's configs state no norm epsilon; their LayerNorms use
/// PyTorch's default, which this is.
const OLMO_NORM_EPS: f64 = 1e-5;

/// GPT-J, GPT-2, GPT-BigCode and BLOOM make the feed-forward this many times
/// the residual width when the config states none, as their Transformers
/// blocks compute it.
const GPT_INNER_PER_HIDDEN: u64 = 4;

/// Candle's Llama config reads Llama's key names. Other families spell the
/// same dimension otherwise (GPT-2's and GPT-J's `n_embd`, `n_layer`,
/// `n_head`, `n_positions`, `n_inner`; BLOOM's `n_embed`; OPT's `ffn_dim`;
/// GPT-NeoX's `rotary_emb_base`; the LayerNorm families' epsilon). Before the
/// config is parsed, the first spelling present is copied under Llama's name;
/// a key Llama's name already holds is left alone. OLMo 1, which states no
/// epsilon, gets the one its norm is defined with; the GPT families without
/// an inner width get four times the residual; GPT-BigCode's `multi_query`
/// becomes one key-value head.
pub(super) fn fill_llama_keys(raw: &mut Value, model_type: &str) {
    // DBRX keeps its attention and feed-forward settings in `attn_config` and
    // `ffn_config`; they are lifted beside the rest before any key is read,
    // never over a key the top level already states.
    if model_type == "dbrx" {
        let nested: Vec<(String, Value)> = ["attn_config", "ffn_config"]
            .iter()
            .filter_map(|section| raw.get(*section).and_then(Value::as_object))
            .flat_map(|section| section.iter().map(|(key, value)| (key.clone(), value.clone())))
            .collect();
        if let Some(object) = raw.as_object_mut() {
            for (key, value) in nested {
                object.entry(key).or_insert(value);
            }
        }
    }
    // Granite 4.0 without experts runs its `shared_mlp` as the feed-forward,
    // `shared_intermediate_size` wide; `intermediate_size` is then unused.
    if model_type == "granitemoehybrid" && whole(raw, "num_local_experts").unwrap_or(0) == 0 {
        let shared = raw.get("shared_intermediate_size").filter(|width| width.is_u64()).cloned();
        if let (Some(width), Some(object)) = (shared, raw.as_object_mut()) {
            object.insert("intermediate_size".to_owned(), width);
        }
    }
    // Zamba2 ties its head to the embeddings whatever `tie_word_embeddings`
    // says (`Zamba2ForCausalLM._tied_weights_keys`), and its checkpoints
    // ship no `lm_head`.
    if model_type == "zamba2" {
        if let Some(object) = raw.as_object_mut() {
            object.insert("tie_word_embeddings".to_owned(), Value::Bool(true));
        }
    }
    // DeciLM states its feed-forward widths per layer and leaves
    // `intermediate_size` null; the widest stands in for the Llama key,
    // which no DeciLM layer reads.
    if model_type == "nemotron-nas" && raw.get("intermediate_size").is_none_or(Value::is_null) {
        let hidden = whole(raw, "hidden_size").unwrap_or(0);
        let widest = raw
            .get("block_configs")
            .and_then(Value::as_array)
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| block.get("ffn"))
                    .filter_map(|ffn| deci_intermediate(ffn, hidden))
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        if let Some(object) = raw.as_object_mut() {
            object.insert("intermediate_size".to_owned(), Value::from(widest));
        }
    }
    // MiMo-V2 states its full-attention layers' key-value heads as
    // `num_key_value_heads` and its sliding-window layers' as
    // `swa_num_key_value_heads`. Ster's base count is the sliding-window
    // one; the full-attention count moves to `num_global_key_value_heads`.
    if model_type.starts_with("mimo_v2") {
        let windowed = raw.get("swa_num_key_value_heads").filter(|value| !value.is_null()).cloned();
        let full = raw.get("num_key_value_heads").cloned();
        if let (Some(windowed), Some(full), Some(object)) = (windowed, full, raw.as_object_mut()) {
            object.insert("num_global_key_value_heads".to_owned(), full);
            object.insert("num_key_value_heads".to_owned(), windowed);
        }
    }
    if model_type == "step3p5" {
        step3p5_keys(raw);
    }
    // LongCat-Flash's `num_layers` stored layers each hold two attention and
    // feed-forward pairs, which Ster runs as two layers.
    if model_type == "longcat_flash" || model_type == "longcat_flash_ngram" {
        let stored = whole(raw, "num_layers");
        if let (Some(stored), Some(object)) = (stored, raw.as_object_mut()) {
            object.insert("num_hidden_layers".to_owned(), Value::from(2 * stored));
        }
    }
    // Cohere2-MoE's `intermediate_size` is each expert's width; its dense
    // prefix layers are `prefix_dense_intermediate_size` wide, which becomes
    // the Llama key. Its norms are RMS norms when it states `rms_norm_eps`
    // and LayerNorms over `layer_norm_eps` otherwise (vLLM's
    // `select_norm_impl`), which `layer_norm` records before the epsilon
    // spellings merge.
    if model_type == "cohere2_moe" {
        let experts = raw.get("intermediate_size").cloned();
        let dense = raw.get("prefix_dense_intermediate_size").filter(|width| !width.is_null()).cloned();
        let layer_norm = raw.get("rms_norm_eps").is_none_or(Value::is_null);
        if let Some(object) = raw.as_object_mut() {
            if let Some(experts) = experts {
                object.entry("moe_intermediate_size").or_insert(experts);
            }
            if let Some(dense) = dense {
                object.insert("intermediate_size".to_owned(), dense);
            }
            object.insert("layer_norm".to_owned(), Value::Bool(layer_norm));
        }
    }
    // Llama 4's `intermediate_size` is each expert's width and
    // `intermediate_size_mlp` its dense layers', which becomes the Llama
    // key.
    if model_type == "llama4_text" {
        let experts = raw.get("intermediate_size").cloned();
        let dense = raw.get("intermediate_size_mlp").filter(|width| !width.is_null()).cloned();
        if let Some(object) = raw.as_object_mut() {
            if let Some(experts) = experts {
                object.entry("moe_intermediate_size").or_insert(experts);
            }
            if let Some(dense) = dense {
                object.insert("intermediate_size".to_owned(), dense);
            }
        }
    }
    if model_type == "nemotron_h_puzzle" {
        nemotron_puzzle_keys(raw);
    }
    if model_type == "chatglm" {
        chatglm_keys(raw);
    }
    // Laguna states its full-attention layers' head count as
    // `num_attention_heads`; Ster's base count is its sliding-window
    // layers', read from `num_attention_heads_per_layer`, and the reading
    // gives the full-attention layers theirs.
    if model_type == "laguna" {
        let windowed = raw
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|kinds| kinds.iter().position(|kind| kind.as_str() == Some("sliding_attention")))
            .and_then(|layer| raw.get("num_attention_heads_per_layer").and_then(|counts| counts.get(layer)).cloned());
        if let (Some(heads), Some(object)) = (windowed, raw.as_object_mut()) {
            object.insert("num_attention_heads".to_owned(), heads);
        }
    }
    // openPangu-Ultra-MoE leaves its router's form to its config class:
    // sigmoid scores, renormalised (`PanguUltraMoEConfig`'s
    // `norm_topk_prob=True`, `MoEGate.forward`).
    if model_type == "pangu_ultra_moe" {
        if let Some(object) = raw.as_object_mut() {
            object.entry("scoring_func").or_insert_with(|| Value::from("sigmoid"));
            object.entry("norm_topk_prob").or_insert(Value::Bool(true));
        }
    }
    let aliases: &[(&str, &[&str])] = &[
        ("rms_norm_eps", &["layer_norm_eps", "norm_epsilon", "norm_eps", "layer_norm_epsilon", "layernorm_epsilon"]),
        ("hidden_size", &["n_embd", "n_embed", "d_model"]),
        ("num_hidden_layers", &["n_layer", "n_layers", "num_layers"]),
        ("num_attention_heads", &["n_head", "n_heads"]),
        ("num_key_value_heads", &["kv_n_heads", "num_attention_groups"]),
        ("head_dim", &["attention_head_dim"]),
        ("max_position_embeddings", &["n_positions", "max_seq_len", "seq_length", "model_max_length"]),
        ("intermediate_size", &["n_inner", "ffn_dim", "ffn_hidden_size"]),
        ("rope_theta", &["rotary_emb_base"]),
        ("num_experts_per_tok", &["moe_k", "moe_top_k", "moe_topk", "num_experts_per_token", "top_k_experts"]),
        // openPangu-Ultra-MoE's names for DeepSeek's latent attention and
        // experts.
        ("kv_lora_rank", &["attention_kv_lora_dim"]),
        ("q_lora_rank", &["attention_q_lora_dim"]),
        ("qk_nope_head_dim", &["attention_qk_dim"]),
        ("qk_rope_head_dim", &["attention_qk_rope_dim"]),
        ("v_head_dim", &["attention_v_dim"]),
        ("n_routed_experts", &["num_routed_experts"]),
        ("n_shared_experts", &["num_shared_experts"]),
        ("first_k_dense_replace", &["num_dense_layers"]),
    ];
    for (llama, spellings) in aliases {
        if raw.get(*llama).is_some_and(|value| !value.is_null()) {
            continue;
        }
        let found = spellings
            .iter()
            .find_map(|key| raw.get(*key).filter(|value| !value.is_null()).cloned());
        if let (Some(value), Some(object)) = (found, raw.as_object_mut()) {
            object.insert((*llama).to_owned(), value);
        }
    }
    let missing = |raw: &Value, key: &str| raw.get(key).map_or(true, Value::is_null);
    let mut defaults: Vec<(&str, Value)> = Vec::new();
    // OLMo 1 and DBRX state no epsilon; their LayerNorms use PyTorch's.
    if matches!(model_type, "olmo" | "dbrx") && missing(raw, "rms_norm_eps") {
        defaults.push(("rms_norm_eps", Value::from(OLMO_NORM_EPS)));
    }
    let four_times = matches!(model_type, "gptj" | "gpt2" | "gpt_bigcode" | "bloom" | "falcon");
    if four_times && missing(raw, "intermediate_size") {
        if let Some(hidden) = raw.get("hidden_size").and_then(Value::as_u64) {
            defaults.push(("intermediate_size", Value::from(GPT_INNER_PER_HIDDEN * hidden)));
        }
    }
    // MPT states its feed-forward as a multiple of the model width.
    if model_type == "mpt" && missing(raw, "intermediate_size") {
        let width = raw.get("hidden_size").and_then(Value::as_u64);
        let ratio = raw.get("expansion_ratio").and_then(Value::as_u64);
        if let (Some(width), Some(ratio)) = (width, ratio) {
            defaults.push(("intermediate_size", Value::from(width * ratio)));
        }
    }
    // LFM2 states its feed-forward as `block_ff_dim`, which Transformers
    // shrinks to two thirds, scales by `block_ffn_dim_multiplier` and rounds
    // up to `block_multiple_of` when `block_auto_adjust_ff_dim` is set.
    if model_type == "lfm2" && missing(raw, "intermediate_size") {
        if let Some(width) = raw.get("block_ff_dim").and_then(Value::as_f64) {
            let width = if flag(raw, "block_auto_adjust_ff_dim") {
                const SWIGLU_SHARE: f64 = 2.0 / 3.0;
                let shrunk = (width * SWIGLU_SHARE).trunc();
                let scaled = match raw.get("block_ffn_dim_multiplier").and_then(Value::as_f64) {
                    Some(multiplier) => (shrunk * multiplier).trunc(),
                    None => shrunk,
                };
                let multiple = raw.get("block_multiple_of").and_then(Value::as_f64).unwrap_or(1.0);
                (scaled / multiple).ceil() * multiple
            } else {
                width
            };
            defaults.push(("intermediate_size", Value::from(width as u64)));
        }
    }
    // Mamba states its inner width as `expand` times the model width when it
    // leaves `intermediate_size` out.
    let state_space_only = matches!(model_type, "mamba" | "falcon_mamba" | "mamba2");
    if state_space_only && missing(raw, "intermediate_size") {
        let width = raw.get("hidden_size").and_then(Value::as_u64);
        let expand = raw.get("expand").and_then(Value::as_u64);
        if let (Some(width), Some(expand)) = (width, expand) {
            defaults.push(("intermediate_size", Value::from(width * expand)));
        }
    }
    // A state-space model has no attention heads and no position table;
    // Candle's config needs both counts, and these stand in, read by nothing.
    if state_space_only {
        if missing(raw, "num_attention_heads") {
            defaults.push(("num_attention_heads", Value::from(1u64)));
        }
        if missing(raw, "max_position_embeddings") {
            defaults.push(("max_position_embeddings", Value::from(1u64)));
        }
    }
    if model_type == "gpt_bigcode" && flag(raw, "multi_query") {
        defaults.push(("num_key_value_heads", Value::from(1u64)));
    }
    // Falcon's new decoder architecture groups query heads under
    // `num_kv_heads`; the older multi-query models have one key-value head.
    if model_type == "falcon" {
        if flag(raw, "new_decoder_architecture") {
            if let Some(groups) = raw.get("num_kv_heads").cloned() {
                defaults.push(("num_key_value_heads", groups));
            }
        } else if flag(raw, "multi_query") {
            defaults.push(("num_key_value_heads", Value::from(1u64)));
        }
    }
    if let Some(object) = raw.as_object_mut() {
        for (key, value) in defaults {
            object.insert(key.to_owned(), value);
        }
    }
}

/// The feed-forward non-linearity the config names, as `hidden_act`,
/// `hidden_activation`, `activation_function` or Nemotron-H's
/// `mlp_hidden_act`; SiLU when it names none.
fn activation(raw: &Value, model_type: &str, path: &Path) -> Result<Activation> {
    let name = text(raw, "hidden_act")
        .or_else(|| text(raw, "hidden_activation"))
        .or_else(|| text(raw, "activation_function"))
        .or_else(|| text(raw, "mlp_hidden_act"));
    match name {
        None | Some("silu") | Some("swish") => Ok(Activation::Silu),
        Some("gelu_pytorch_tanh") | Some("gelu_new") | Some("gelu_fast") => Ok(Activation::GeluTanh),
        Some("gelu") => Ok(Activation::Gelu),
        Some("relu2") => Ok(Activation::Relu2),
        Some("relu") => Ok(Activation::Relu),
        Some("xielu") => Ok(Activation::Xielu),
        Some(other) => bail!(
            "{} declares hidden_act {other:?} for its {model_type} feed-forward; Ster implements silu, gelu, gelu_pytorch_tanh, gelu_new, gelu_fast, relu, relu2 and xielu",
            path.display()
        ),
    }
}

/// A `rope_scaling` Candle's Llama config cannot read (anything but Llama
/// 3's), taken out of the config before it is parsed so Ster can apply it
/// itself. Returns what was taken.
///
/// Configs written by Transformers 5 state the rotation in one
/// `rope_parameters` object; its `rope_theta` is read as the base and, unless
/// its `rope_type` is `default`, the object as the scaling, never over keys
/// the config states at the top level. Gemma 3's and 4's state one object per
/// layer type: the `full_attention` one is the model's rotation, and the
/// `sliding_attention` one's base is `rope_local_base_freq`.
pub(super) fn take_rope_scaling(raw: &mut Value, path: &Path) -> Result<Option<Value>> {
    let per_type = raw.get("rope_parameters").filter(|value| value.get("full_attention").is_some()).cloned();
    if let Some(per_type) = per_type {
        if let Some(sliding) = per_type.get("sliding_attention").filter(|value| value.is_object()) {
            let kind = scaling_kind(sliding);
            if !matches!(kind, "default" | "") {
                bail!(
                    "{} scales its sliding_attention rotation ({kind:?}); Ster scales only the full-attention layers' rotation",
                    path.display()
                );
            }
            if let (Some(theta), Some(object)) =
                (sliding.get("rope_theta").filter(|theta| theta.is_number()).cloned(), raw.as_object_mut())
            {
                object.entry("rope_local_base_freq").or_insert(theta);
            }
            // The sliding-window layers' rotated share, which a family that
            // gives each kind its own (Laguna) reads.
            if let (Some(share), Some(object)) =
                (sliding.get("partial_rotary_factor").filter(|share| share.is_number()).cloned(), raw.as_object_mut())
            {
                object.entry("local_partial_rotary_factor").or_insert(share);
            }
        }
        // Chunked layers (Rnj-1.5) rotate by the global table, so their
        // stated rotation must be the full-attention one.
        if let Some(chunked) = per_type.get("chunked_attention").filter(|value| value.is_object()) {
            if per_type.get("full_attention") != Some(chunked) {
                bail!(
                    "{} rotates its chunked_attention layers unlike its full_attention layers; Ster rotates both by the full-attention rotation",
                    path.display()
                );
            }
        }
        if let (Some(full), Some(object)) = (per_type.get("full_attention").cloned(), raw.as_object_mut()) {
            object.insert("rope_parameters".to_owned(), full);
        }
    }
    if let Some(parameters) = raw.get("rope_parameters").filter(|value| value.is_object()).cloned() {
        let scaled = !matches!(scaling_kind(&parameters), "default" | "");
        if let Some(object) = raw.as_object_mut() {
            if let Some(theta) = parameters.get("rope_theta").filter(|theta| theta.is_number()) {
                object.entry("rope_theta").or_insert_with(|| theta.clone());
            }
            let unstated = object.get("rope_scaling").map_or(true, Value::is_null);
            if scaled && unstated {
                object.insert("rope_scaling".to_owned(), parameters);
            }
        }
    }
    let Some(scaling) = raw.get("rope_scaling").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    match scaling_kind(scaling) {
        "llama3" => Ok(None),
        "default" | "linear" | "longrope" | "yarn" | "telechat3-yarn" | "proportional" | "deepseek_yarn" => Ok(raw
            .as_object_mut()
            .and_then(|object| object.remove("rope_scaling"))),
        // HunYuan states `dynamic` with an `alpha`: a fixed NTK-aware base,
        // `rope_theta * alpha^(d / (d - 2))` over the head width `d`, the
        // same at every length.
        "dynamic" if scaling.get("alpha").and_then(Value::as_f64).is_some() => {
            let alpha = scaling.get("alpha").and_then(Value::as_f64).unwrap_or(1.0);
            let width = whole(raw, "head_dim").or_else(|| {
                Some(whole(raw, "hidden_size")? / whole(raw, "num_attention_heads")?.max(1))
            });
            let Some(width) = width.filter(|width| *width > 2) else {
                bail!(
                    "{} declares dynamic rope_scaling with alpha but no head width to scale by",
                    path.display()
                );
            };
            let theta = number(raw, "rope_theta").unwrap_or(DEFAULT_ROPE_THETA);
            let scaled = theta * alpha.powf(width as f64 / (width as f64 - 2.0));
            if let Some(object) = raw.as_object_mut() {
                object.remove("rope_scaling");
                object.insert("rope_theta".to_owned(), Value::from(scaled));
            }
            Ok(None)
        }
        // Without an alpha, dynamic NTK scaling changes the rotary base only
        // once a sequence outgrows `max_position_embeddings`, which Ster
        // never runs; within it this is the plain rotation.
        "dynamic" => {
            if let Some(object) = raw.as_object_mut() {
                object.remove("rope_scaling");
            }
            Ok(None)
        }
        other => bail!(
            "{} declares rope_scaling {other:?}; Ster implements llama3, linear, longrope, yarn, deepseek_yarn, telechat3-yarn, proportional and dynamic rotary scaling",
            path.display()
        ),
    }
}

/// How many earlier chunks Rnj-1.5's chunked layers see (vLLM's `rnj1.py`,
/// `chunk_lookback = 1 if self.is_chunked`).
const RNJ1_CHUNK_LOOKBACK: usize = 1;

/// The rotary base a config without `rope_theta` uses, Llama's and Candle's
/// default.
const DEFAULT_ROPE_THETA: f64 = 10_000.0;

fn scaling_kind(scaling: &Value) -> &str {
    scaling
        .get("rope_type")
        .or_else(|| scaling.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// The scaling [`take_rope_scaling`] took, over `rotary_dim` components.
fn rope_scaling(
    scaling: Option<&Value>,
    rotary_dim: usize,
    raw: &Value,
    llama: &LlamaConfig,
    path: &Path,
) -> Result<RopeScaling> {
    let Some(scaling) = scaling else {
        return Ok(RopeScaling::None);
    };
    match scaling_kind(scaling) {
        "linear" => {
            let Some(factor) = scaling.get("factor").and_then(Value::as_f64) else {
                bail!("{} declares linear rope_scaling with no factor", path.display());
            };
            Ok(RopeScaling::Linear(factor as f32))
        }
        "longrope" => {
            let factors = |key: &str| -> Result<Vec<f32>> {
                let values: Option<Vec<f32>> = scaling
                    .get(key)
                    .and_then(Value::as_array)
                    .and_then(|list| list.iter().map(|v| v.as_f64().map(|v| v as f32)).collect());
                match values {
                    Some(values) if values.len() == rotary_dim / 2 => Ok(values),
                    _ => bail!(
                        "{} declares longrope {key} that is not {} numbers, one per rotated frequency",
                        path.display(),
                        rotary_dim / 2
                    ),
                }
            };
            let (short, long) = (factors("short_factor")?, factors("long_factor")?);
            // Phi-3 states the original context at the top level and derives
            // the factor from it, as Transformers does.
            let maximum = llama.max_position_embeddings;
            let (original, factor) = match whole(raw, "original_max_position_embeddings") {
                Some(original) => (original, maximum as f64 / original as f64),
                None => (maximum, scaling.get("factor").and_then(Value::as_f64).unwrap_or(1.0)),
            };
            let attention = match scaling.get("attention_factor").and_then(Value::as_f64) {
                Some(attention) => attention,
                None if factor <= 1.0 => 1.0,
                None => (1.0 + factor.ln() / (original as f64).ln()).sqrt(),
            };
            // PhiMoE states each table's magnitude itself.
            let magnitude = |key: &str| scaling.get(key).and_then(Value::as_f64).unwrap_or(attention);
            Ok(RopeScaling::LongRope {
                short,
                long,
                original,
                short_attention: magnitude("short_mscale") as f32,
                long_attention: magnitude("long_mscale") as f32,
            })
        }
        // TeleChat3's `telechat3-yarn` is YaRN with a gentler temperature
        // slope (its `_compute_telechat_yarn_parameters`); vLLM's
        // `deepseek_yarn` (Sarvam) is YaRN with DeepSeek's `mscale` ratio.
        kind @ ("yarn" | "telechat3-yarn" | "deepseek_yarn") => {
            let slope = if kind == "telechat3-yarn" { TELECHAT3_YARN_SLOPE } else { YARN_SLOPE };
            let Some(factor) = scaling.get("factor").and_then(Value::as_f64) else {
                bail!("{} declares yarn rope_scaling with no factor", path.display());
            };
            let original = scaling
                .get("original_max_position_embeddings")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(llama.max_position_embeddings);
            let stated = |key: &str| scaling.get(key).and_then(Value::as_f64).filter(|v| *v != 0.0);
            // Transformers' `_compute_yarn_parameters`: the cos and sin
            // magnitude is `mscale / mscale_all_dim` when both are stated,
            // otherwise the stated `attention_factor`, otherwise YaRN's own.
            let attention = match (stated("mscale"), stated("mscale_all_dim")) {
                (Some(mscale), Some(all)) => yarn_mscale(factor, mscale, slope) / yarn_mscale(factor, all, slope),
                _ => stated("attention_factor").unwrap_or_else(|| yarn_mscale(factor, 1.0, slope)),
            };
            Ok(RopeScaling::Yarn {
                factor: factor as f32,
                original,
                beta_fast: stated("beta_fast").unwrap_or(YARN_BETA_FAST) as f32,
                beta_slow: stated("beta_slow").unwrap_or(YARN_BETA_SLOW) as f32,
                attention: attention as f32,
            })
        }
        "proportional" => Ok(proportional(scaling, rotary_dim)),
        _ => Ok(RopeScaling::None),
    }
}

/// Gemma 4's proportional rotation over `width` components, as
/// Transformers' `_compute_proportional_rope_parameters` computes it: the
/// first `partial_rotary_factor · width / 2` frequency pairs, rounded down,
/// rotate at `rope_theta^(-2i / width)`, the rest are zero, and every
/// frequency is divided by `factor`.
fn proportional(scaling: &Value, width: usize) -> RopeScaling {
    let share = scaling.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0);
    RopeScaling::Proportional {
        rotated: (share * width as f64 / 2.0).floor() as usize,
        factor: scaling.get("factor").and_then(Value::as_f64).unwrap_or(1.0) as f32,
    }
}

/// YaRN's default rotation counts bounding the interpolated band (Peng et
/// al., 2023; Transformers' `beta_fast` and `beta_slow`).
const YARN_BETA_FAST: f64 = 32.0;
const YARN_BETA_SLOW: f64 = 1.0;

/// YaRN's temperature slope (Peng et al., 2023; Transformers'
/// `get_mscale`), and TeleChat3's (`get_mscale` in its
/// `modeling_telechat3.py`).
const YARN_SLOPE: f64 = 0.1;
const TELECHAT3_YARN_SLOPE: f64 = 0.07;

/// YaRN's attention temperature for a context stretched by `factor`:
/// `slope * mscale * ln(factor) + 1`, or one when nothing is stretched.
fn yarn_mscale(factor: f64, mscale: f64, slope: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        slope * mscale * factor.ln() + 1.0
    }
}

/// What `model_type` adds to the Llama block, from the config's own keys and
/// the `rope_scaling` [`take_rope_scaling`] took out of them.
pub(super) fn family(
    model_type: &str,
    raw: &Value,
    scaling: Option<&Value>,
    llama: &LlamaConfig,
    path: &Path,
) -> Result<Architecture> {
    let layers = llama.num_hidden_layers;
    let mut architecture = Architecture::llama(
        llama.hidden_size,
        llama.num_attention_heads,
        llama.rms_norm_eps,
    );
    if let Some(head_dim) = whole(raw, "head_dim") {
        architecture.head_dim = head_dim;
        architecture.score_divisor = (head_dim as f64).sqrt();
    }
    // Multi-head latent attention (DeepSeek-V2 and V3, MiniCPM3) states its
    // bottlenecks and head parts; a head is its position-free part plus its
    // rotated part, and only the rotated part rotates.
    if let Some(key_value_rank) = whole(raw, "kv_lora_rank") {
        let part = |key: &str| -> Result<usize> {
            whole(raw, key).with_context(|| {
                format!("{} declares kv_lora_rank without {key}", path.display())
            })
        };
        let latent = LatentAttention {
            query_rank: whole(raw, "q_lora_rank"),
            key_value_rank,
            unrotated: part("qk_nope_head_dim")?,
            rotated: part("qk_rope_head_dim")?,
            value: part("v_head_dim")?,
        };
        architecture.head_dim = latent.unrotated + latent.rotated;
        architecture.score_divisor = (architecture.head_dim as f64).sqrt();
        architecture.latent = Some(latent);
    }
    // The rotated share of each head: `partial_rotary_factor` (or GPT-NeoX's
    // `rotary_pct`) as a fraction, which Transformers truncates, GPT-J's
    // `rotary_dim` as a width, or latent attention's rotated part.
    let head_dim = architecture.head_dim;
    let fraction = number(raw, "partial_rotary_factor").or_else(|| number(raw, "rotary_pct"));
    architecture.rotary_dim = match (architecture.latent, fraction, whole(raw, "rotary_dim")) {
        (Some(latent), _, _) => latent.rotated,
        (None, Some(factor), _) => (head_dim as f64 * factor) as usize,
        (None, None, Some(width)) => width,
        (None, None, None) => head_dim,
    };
    let rotary_dim = architecture.rotary_dim;
    if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        bail!(
            "{} rotates {rotary_dim} of {head_dim} components per head (partial_rotary_factor, rotary_pct or rotary_dim); the rotated width must be even, above zero and at most the head",
            path.display()
        );
    }
    architecture.rope_scaling = rope_scaling(scaling, architecture.rotary_dim, raw, llama, path)?;
    // DeepSeek sharpens the scores of a YaRN-stretched model by the square
    // of the temperature `mscale_all_dim` gives.
    if architecture.latent.is_some() {
        let all = scaling
            .and_then(|scaling| scaling.get("mscale_all_dim"))
            .and_then(Value::as_f64)
            .filter(|value| *value != 0.0);
        if let (RopeScaling::Yarn { factor, .. }, Some(all)) = (&architecture.rope_scaling, all) {
            let temperature = yarn_mscale(f64::from(*factor), all, YARN_SLOPE);
            architecture.score_divisor /= temperature * temperature;
        }
    }
    // Gemma's configs name `gelu` but Transformers runs the tanh
    // approximation for every Gemma, so the family decides, not the key.
    if !model_type.starts_with("gemma") && model_type != "bloom" {
        architecture.activation = activation(raw, model_type, path)?;
    }
    match model_type {
        "ministral3" => {
            // Ministral 3 (the text decoder of `mistral3`): Mistral's block
            // with Llama 4's query temperature, `llama_4_scaling_beta` past
            // every `original_max_position_embeddings` positions, both
            // stated beside its rotation (vLLM's `llama_4_scaling`).
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.sliding_window = whole(raw, "sliding_window");
            if architecture.sliding_window.is_some() {
                architecture.sliding_layers = every_layer(layers, path)?;
            }
            architecture.query_temperature = llama4_temperature(raw, scaling, layers, path)?;
        }
        "llama" | "mistral" | "mixtral" | "phi3" | "phimoe" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            // Mistral's own weights rotate adjacent pairs (their reading
            // states `rope_interleave`).
            architecture.interleaved_rotary = flag(raw, "rope_interleave");
            if model_type == "phi3" {
                architecture.qkv_layout = QkvLayout::Stacked;
            }
            architecture.fused_feed_forward = model_type == "phi3";
            // Mistral v0.2 and later publish `sliding_window: null`, which is
            // full attention on every layer; Phi-3 states a window it applies
            // on every layer.
            if model_type != "llama" {
                architecture.sliding_window = whole(raw, "sliding_window");
                if architecture.sliding_window.is_some() {
                    architecture.sliding_layers = every_layer(layers, path)?;
                }
            }
            if model_type == "mixtral" {
                architecture.experts = Some(experts(
                    raw,
                    "num_local_experts",
                    "intermediate_size",
                    true,
                    ExpertLayout::Mixtral,
                    0,
                    path,
                )?);
            }
            if model_type == "phimoe" {
                // PhiMoE: Mixtral's tensors under LayerNorms with bias, a
                // biased vocabulary projection when `lm_head_bias` says so,
                // and SparseMixer routing, which Transformers defines for
                // two experts per token.
                architecture.norm = NormKind::Layer { bias: true };
                architecture.lm_head_bias = flag(raw, "lm_head_bias");
                let mut routed = experts(
                    raw,
                    "num_local_experts",
                    "intermediate_size",
                    false,
                    ExpertLayout::Mixtral,
                    0,
                    path,
                )?;
                if routed.top_k != SPARSE_MIXER_EXPERTS {
                    bail!(
                        "{} routes {} experts per token; PhiMoE's SparseMixer chooses exactly {SPARSE_MIXER_EXPERTS}",
                        path.display(),
                        routed.top_k
                    );
                }
                let Some(jitter) = number(raw, "router_jitter_noise") else {
                    bail!(
                        "{} states no router_jitter_noise; PhiMoE's SparseMixer needs it to bound the experts it weighs",
                        path.display()
                    );
                };
                routed.scoring = Scoring::SparseMixer { jitter: jitter as f32 };
                architecture.experts = Some(routed);
            }
        }
        "qwen2" | "qwen3" | "qwen2_moe" | "qwen3_moe" | "qwen3_next" | "mimo" | "mellum" | "qwen3_5_text"
        | "qwen3_5_moe_text" => {
            // MiMo is Qwen2 with next-token-prediction layers
            // (`model.mtp_layers`) that a single forward never reads.
            if model_type.starts_with("qwen2") || model_type == "mimo" {
                architecture.query_key_value_bias = true;
            } else {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.query_key_value_bias = flag(raw, "attention_bias");
                architecture.output_bias = architecture.query_key_value_bias;
            }
            if flag(raw, "use_sliding_window") {
                architecture.sliding_window = whole(raw, "sliding_window");
                let from = whole(raw, "max_window_layers").unwrap_or(0);
                architecture.sliding_layers = every_layer(layers, path)? & !every_layer(from.min(layers), path)?;
            }
            // Qwen3.5 (`qwen3_5_text`, `qwen3_5_moe_text`) is Qwen3-Next with
            // its linear attention's projections stored apart.
            let qwen35 = model_type.starts_with("qwen3_5");
            if model_type == "qwen3_next" || qwen35 {
                // Qwen3-Next: norms that store their scale as an offset from
                // one, a sigmoid gate on attention's output from `q_proj`
                // (under `attn_output_gate`, on by default), and gated
                // delta-rule linear attention on every layer `layer_types`
                // calls `linear_attention` (without it, on every layer but
                // each `full_attention_interval`-th).
                fits(layers, path)?;
                let linear = match raw.get("layer_types").and_then(Value::as_array) {
                    Some(types) => {
                        if types.len() != layers {
                            bail!("{} lists {} layer_types for {layers} layers", path.display(), types.len());
                        }
                        let mut linear = 0u128;
                        for (layer, kind) in types.iter().enumerate() {
                            match kind.as_str() {
                                Some("linear_attention") => linear |= 1u128 << layer,
                                Some("full_attention") => {}
                                other => bail!(
                                    "{} names layer {layer} {other:?}; a Qwen3-Next layer is linear_attention or full_attention",
                                    path.display()
                                ),
                            }
                        }
                        linear
                    }
                    None => {
                        let Some(interval) = whole(raw, "full_attention_interval").filter(|n| *n > 0) else {
                            bail!(
                                "{} declares a Qwen3-Next model with neither layer_types nor full_attention_interval",
                                path.display()
                            );
                        };
                        (0..layers)
                            .filter(|layer| (layer + 1) % interval != 0)
                            .fold(0u128, |set, layer| set | (1u128 << layer))
                    }
                };
                let size = |key: &str| -> Result<usize> {
                    whole(raw, key)
                        .filter(|size| *size > 0)
                        .with_context(|| format!("{} declares a Qwen3-Next model without {key}", path.display()))
                };
                architecture.norm_offset = true;
                architecture.output_gate = raw.get("attn_output_gate").and_then(Value::as_bool).unwrap_or(true);
                architecture.delta_rule = Some(DeltaRuleSpec {
                    key_heads: size("linear_num_key_heads")?,
                    value_heads: size("linear_num_value_heads")?,
                    key_dim: size("linear_key_head_dim")?,
                    value_dim: size("linear_value_head_dim")?,
                    kernel: size("linear_conv_kernel_dim")?,
                    layers: linear,
                    form: if qwen35 { DeltaRuleForm::Qwen35 } else { DeltaRuleForm::Qwen3Next },
                    negative_eigenvalues: false,
                    decay_floor: None,
                });
            }
            // Mellum is Qwen3-MoE with sliding-window layers, a rotation per
            // layer kind, and dense layers where `mlp_layer_types` says so.
            if model_type.ends_with("_moe") || model_type.ends_with("_moe_text") || model_type == "qwen3_next" || model_type == "mellum" {
                // Qwen3-Next and Qwen3.5 renormalise unless told not to
                // (vLLM's `getattr(config, "norm_topk_prob", True)`); Qwen2-
                // and Qwen3-MoE only when told to.
                let normalize = raw
                    .get("norm_topk_prob")
                    .and_then(Value::as_bool)
                    .unwrap_or(model_type == "qwen3_next" || qwen35);
                let mut routed = experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    normalize,
                    ExpertLayout::Qwen,
                    qwen_dense_layers(raw, layers, path)?,
                    path,
                )?;
                if model_type != "qwen3_moe" {
                    routed.shared = whole(raw, "shared_expert_intermediate_size")
                        .map(|intermediate| SharedExpert {
                            intermediate,
                            module: "mlp.shared_expert",
                            gated: true,
                            form: SharedForm::GateUpDown,
                        });
                }
                architecture.experts = Some(routed);
            }
        }
        "granite" | "granitemoe" | "granitemoehybrid" | "granite_swa" | "granitemoe_swa" | "granitemoeshared"
        | "hyperclovax" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.embedding_multiplier = number(raw, "embedding_multiplier");
            architecture.residual_multiplier = number(raw, "residual_multiplier");
            if let Some(multiplier) = number(raw, "attention_multiplier") {
                architecture.score_divisor = 1.0 / multiplier;
            }
            // HyperCLOVAX multiplies its logits by `logits_scaling` where
            // Granite divides by it, and under `use_post_norm` (on unless
            // stated off) normalises each sublayer's output (`post_norm1`,
            // `post_norm2`) before the scaled residual add.
            architecture.logits_multiplier = if model_type == "hyperclovax" {
                number(raw, "logits_scaling")
            } else {
                number(raw, "logits_scaling").map(|scale| 1.0 / scale)
            };
            if model_type == "hyperclovax" && raw.get("use_post_norm").and_then(Value::as_bool) != Some(false) {
                architecture.output_norms = true;
                architecture.names = Names::HYPERCLOVAX;
            }
            if matches!(model_type, "granitemoe" | "granitemoe_swa" | "granitemoeshared") {
                // GraniteMoE takes the softmax over the top-k logits, which is
                // the full softmax renormalised over the chosen experts.
                let mut routed =
                    experts(raw, "num_local_experts", "intermediate_size", true, ExpertLayout::Granite, 0, path)?;
                // GraniteMoeShared and GraniteMoeSWA add `shared_mlp`,
                // `shared_intermediate_size` wide, to every layer's experts
                // when that width is above zero.
                if model_type != "granitemoe" {
                    routed.shared = whole(raw, "shared_intermediate_size").filter(|width| *width > 0).map(
                        |intermediate| SharedExpert {
                            intermediate,
                            module: "shared_mlp",
                            gated: false,
                            form: SharedForm::Stacked,
                        },
                    );
                }
                architecture.experts = Some(routed);
            }
            if model_type.ends_with("_swa") {
                granite_windows(raw, layers, llama, &mut architecture, path)?;
            }
            if model_type == "granitemoehybrid" {
                granite_hybrid(raw, layers, &mut architecture, path)?;
            }
        }
        "olmoe" => {
            architecture.clip_qkv = number(raw, "clip_qkv");
            architecture.query_key_norm = QueryKeyNorm::Full;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.experts = Some(experts(
                raw,
                "num_experts",
                "intermediate_size",
                flag(raw, "norm_topk_prob"),
                ExpertLayout::Qwen,
                0,
                path,
            )?);
        }
        "stablelm" => {
            architecture.norm = NormKind::Layer { bias: true };
            architecture.query_key_value_bias = flag(raw, "use_qkv_bias");
            architecture.parallel = flag(raw, "use_parallel_residual");
            if flag(raw, "qk_layernorm") {
                architecture.query_key_norm = QueryKeyNorm::HeadModules;
            }
        }
        "starcoder2" => {
            let bias = raw.get("use_bias").and_then(Value::as_bool).unwrap_or(true);
            architecture.norm = NormKind::Layer { bias: true };
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::STARCODER2;
            architecture.sliding_window = whole(raw, "sliding_window");
            if architecture.sliding_window.is_some() {
                architecture.sliding_layers = every_layer(layers, path)?;
            }
        }
        "cohere" | "cohere2" => {
            architecture.norm = NormKind::Layer { bias: false };
            architecture.parallel = true;
            architecture.interleaved_rotary = true;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.logits_multiplier = number(raw, "logit_scale");
            if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::HeadWeights;
            }
            if model_type == "cohere2" {
                // Every `sliding_window_pattern`-th layer attends globally and
                // applies no rotary embedding; the rest are local and rotate.
                architecture.sliding_window = whole(raw, "sliding_window");
                if let Some(pattern) = whole(raw, "sliding_window_pattern").filter(|p| *p > 0) {
                    fits(layers, path)?;
                    architecture.sliding_layers = (0..layers)
                        .filter(|layer| (layer + 1) % pattern != 0)
                        .fold(0, |set, layer| set | (1u128 << layer));
                }
            }
        }
        "nemotron" => {
            // LayerNorm1P: a LayerNorm whose stored scale is an offset from one.
            architecture.norm = NormKind::Layer { bias: true };
            architecture.norm_offset = true;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::UP_DOWN;
        }
        "phi" => {
            if flag(raw, "qk_layernorm") {
                bail!(
                    "{} declares qk_layernorm; Ster implements Phi without query and key norms",
                    path.display()
                );
            }
            architecture.norm = NormKind::Layer { bias: true };
            architecture.parallel = true;
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.lm_head_bias = true;
            architecture.names = Names::PHI;
        }
        "nemotron-nas" => {
            // DeciLM (Llama-3.3-Nemotron Super and Ultra): Llama's block,
            // each layer's own `block_configs` entry giving its key-value
            // heads (`num_attention_heads / n_heads_in_group`) and
            // feed-forward width (`ffn_mult`), either half possibly a
            // no-op. Linear replacements, attention windows and sinks, and
            // sparsified layers are refused.
            architecture.query_key_value_bias = flag(raw, "attention_bias") || flag(raw, "bias");
            architecture.output_bias = architecture.query_key_value_bias;
            if let Some(bias) = raw.get("qkv_bias").and_then(Value::as_bool) {
                architecture.query_key_value_bias = bias;
            }
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.layer_plans = Some(deci_plans(raw, layers, llama, path)?);
        }
        "nanbeige" => {
            // Nanbeige: Llama's block, per-head `q_layernorm` and
            // `k_layernorm` under `qk_layernorm`, and the stored layers run
            // `num_loops` times, the final norm closing every pass unless
            // `skip_loop_final_norm`. Its n-gram embeddings, hyper-connections,
            // split loops, shared loop caches and depth attention are
            // refused.
            for key in [
                "enable_hyper_connection",
                "enable_mhc",
                "enable_double_loop_split",
                "loop_share_kv",
                "enable_depth_attention",
            ] {
                if flag(raw, key) {
                    bail!("{} turns on Nanbeige's {key}, which Ster does not implement", path.display());
                }
            }
            if raw.get("ngram_vocab_size_ratio").is_some_and(|value| !value.is_null()) {
                bail!(
                    "{} declares Nanbeige's n-gram embeddings (ngram_vocab_size_ratio), which Ster does not implement",
                    path.display()
                );
            }
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            if flag(raw, "qk_layernorm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.names = Names::NANBEIGE;
            }
            let count = whole(raw, "num_loops").unwrap_or(1).max(1);
            if count > 1 {
                architecture.loops = Some(Loops {
                    physical: layers,
                    count,
                    norm_between: !flag(raw, "skip_loop_final_norm"),
                    gate_window: None,
                });
            }
        }
        "iquestloopcoder" => {
            // IQuest-LoopCoder: Llama's block with `mlp_bias`, its stored
            // layers run `loop_num` times; each later pass mixes global
            // attention over the first pass's keys and values with local
            // attention over its own within `loop_window_size`, gated per
            // head by `model.gate_projections.{layer}` of the rotated query.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            let count = whole(raw, "loop_num").unwrap_or(LOOP_CODER_LOOPS).max(1);
            let window = whole(raw, "loop_window_size").unwrap_or(LOOP_CODER_WINDOW);
            if count > 1 {
                architecture.loops = Some(Loops { physical: layers, count, norm_between: false, gate_window: Some(window) });
            }
        }
        "hy_v3" => {
            // HY V3 (Hy3): per-head query and key norms, a dense
            // feed-forward on the layers `mlp_layer_types` calls `dense` (or
            // the first `first_k_dense_replace`), and elsewhere `num_experts`
            // experts chosen by sigmoid score plus `mlp.expert_bias`,
            // renormalised under `route_norm`, scaled by
            // `router_scaling_factor`, beside `num_shared_experts` shared
            // experts in `mlp.shared_mlp`.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            fits(layers, path)?;
            let dense = if raw.get("mlp_layer_types").is_some_and(Value::is_array) {
                qwen_dense_layers(raw, layers, path)?
            } else {
                let first = whole(raw, "first_k_dense_replace").unwrap_or(0).min(layers);
                (0..first).fold(0u128, |set, layer| set | (1u128 << layer))
            };
            let normalize = raw.get("route_norm").and_then(Value::as_bool).unwrap_or(true);
            let mut routed =
                experts(raw, "num_experts", "moe_intermediate_size", normalize, ExpertLayout::HyV3, dense, path)?;
            routed.scoring = Scoring::Sigmoid;
            routed.selection_bias = Some("mlp.expert_bias");
            routed.routed_scale = number(raw, "router_scaling_factor");
            routed.shared = whole(raw, "num_shared_experts").filter(|shared| *shared > 0).map(|shared| SharedExpert {
                intermediate: shared * routed.intermediate,
                module: "mlp.shared_mlp",
                gated: false,
                form: SharedForm::GateUpDown,
            });
            architecture.experts = Some(routed);
        }
        "olmo_hybrid" => {
            // OLMo Hybrid: Gated DeltaNet (`linear_attn`) in a pre-norm block
            // (`input_layernorm`, the mixer, `post_attention_layernorm`, the
            // feed-forward) on the layers `layer_types` calls
            // `linear_attention` — each fourth from the fourth attending
            // when it lists none — and OLMo 3's post-norm attention block
            // with whole-projection query and key norms on the rest, rotated
            // only when `rope_parameters` states a `rope_theta`.
            fits(layers, path)?;
            let linear = match raw.get("layer_types").and_then(Value::as_array) {
                Some(types) => {
                    if types.len() != layers {
                        bail!("{} lists {} layer_types for {layers} layers", path.display(), types.len());
                    }
                    let mut linear = 0u128;
                    for (layer, kind) in types.iter().enumerate() {
                        match kind.as_str() {
                            Some("linear_attention") => linear |= 1u128 << layer,
                            Some("full_attention") => {}
                            other => bail!(
                                "{} names layer {layer} {other:?}; an OLMo Hybrid layer is linear_attention or full_attention",
                                path.display()
                            ),
                        }
                    }
                    linear
                }
                None => {
                    let attending = |layer: usize| layer % OLMO_HYBRID_FULL_EVERY == OLMO_HYBRID_FULL_EVERY - 1;
                    let mut linear =
                        (0..layers).filter(|layer| !attending(*layer)).fold(0u128, |set, layer| set | (1u128 << layer));
                    if layers > 0 && (0..layers).all(|layer| !attending(layer)) {
                        linear &= !(1u128 << (layers - 1));
                    }
                    linear
                }
            };
            let size = |key: &str| -> Result<usize> {
                whole(raw, key)
                    .filter(|size| *size > 0)
                    .with_context(|| format!("{} declares an OLMo Hybrid model without {key}", path.display()))
            };
            architecture.query_key_norm = QueryKeyNorm::Full;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.pre_norms = false;
            architecture.output_norms = true;
            architecture.delta_rule = Some(DeltaRuleSpec {
                key_heads: size("linear_num_key_heads")?,
                value_heads: size("linear_num_value_heads")?,
                key_dim: size("linear_key_head_dim")?,
                value_dim: size("linear_value_head_dim")?,
                kernel: size("linear_conv_kernel_dim")?,
                layers: linear,
                form: DeltaRuleForm::OlmoHybrid,
                negative_eigenvalues: raw.get("linear_allow_neg_eigval").and_then(Value::as_bool).unwrap_or(true),
                decay_floor: None,
            });
            let stated_theta = |object: Option<&Value>| object.and_then(|value| value.get("rope_theta")).is_some_and(Value::is_number);
            if !stated_theta(raw.get("rope_parameters")) && !stated_theta(Some(raw)) {
                architecture.positions = Positions::None;
            }
        }
        "olmo2" | "olmo3" | "flex_olmo" => {
            architecture.query_key_norm = QueryKeyNorm::Full;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.pre_norms = false;
            architecture.output_norms = true;
            if model_type == "olmo3" {
                architecture.sliding_window = whole(raw, "sliding_window");
            }
            // FlexOlmo: OLMo 2's block with OLMoE's experts as its
            // feed-forward.
            if model_type == "flex_olmo" {
                architecture.experts = Some(experts(
                    raw,
                    "num_experts",
                    "intermediate_size",
                    flag(raw, "norm_topk_prob"),
                    ExpertLayout::Qwen,
                    0,
                    path,
                )?);
            }
        }
        "smollm3" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            if let Some(flags) = raw.get("no_rope_layers").and_then(Value::as_array) {
                architecture.unrotated_layers = unrotated(flags, layers, path)?;
            }
            if flag(raw, "use_sliding_window") {
                architecture.sliding_window = whole(raw, "sliding_window");
                architecture.sliding_layers = every_layer(layers, path)?;
            }
        }
        "olmo" => {
            // OLMo 1's LayerNorm stores neither a scale nor a bias.
            architecture.norm = NormKind::Bare;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.clip_qkv = number(raw, "clip_qkv");
        }
        "dbrx" => {
            // DBRX: LayerNorms without bias, one `Wqkv` clipped to
            // `clip_qkv`, and a mixture of `moe_num_experts` experts whose
            // top `moe_top_k` weights are divided by their p-norm, p being
            // `moe_normalize_expert_weights`.
            let normalize = match raw.get("moe_normalize_expert_weights") {
                None | Some(Value::Null) => false,
                Some(power) if power.as_f64() == Some(1.0) => true,
                Some(other) => bail!(
                    "{} normalises expert weights by their {other}-norm; Ster implements the 1-norm (moe_normalize_expert_weights 1) or none",
                    path.display()
                ),
            };
            architecture.names = Names::DBRX;
            architecture.norm = NormKind::Layer { bias: false };
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.clip_qkv = number(raw, "clip_qkv");
            architecture.experts = Some(experts(
                raw,
                "moe_num_experts",
                "intermediate_size",
                normalize,
                ExpertLayout::Dbrx,
                0,
                path,
            )?);
        }
        "exaone4" => {
            // OLMo 2's post-norm block with per-head query and key norms.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.pre_norms = false;
            architecture.output_norms = true;
            architecture.sliding_window = whole(raw, "sliding_window");
            if let Some(pattern) = whole(raw, "sliding_window_pattern").filter(|p| *p > 0) {
                fits(layers, path)?;
                architecture.sliding_layers = (0..layers)
                    .filter(|layer| (layer + 1) % pattern != 0)
                    .fold(0, |set, layer| set | (1u128 << layer));
            }
        }
        "apertus" => {
            // Apertus: per-head query and key norms, `attention_layernorm`
            // and `feedforward_layernorm`, and a plain up-activation-down
            // feed-forward whose xIELU keeps its parameters in `mlp.act_fn`.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::APERTUS;
        }
        "exaone_moe" => {
            // K-EXAONE: EXAONE 4's attention (per-head query and key norms,
            // and no rotation on its full-attention layers beside
            // sliding-window ones) in a pre-norm block, with DeepSeek-V3's
            // sigmoid router and `num_shared_experts` shared experts on the
            // layers `mlp_layer_types` calls `sparse`.
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.sliding_window = whole(raw, "sliding_window");
            architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
        }
        "internlm3" => {
            architecture.query_key_value_bias = flag(raw, "qkv_bias");
            architecture.output_bias = flag(raw, "bias");
        }
        "TeleFLM" => {
            // Tele-FLM: Llama's block; under `use_mup` the embeddings are
            // multiplied by `input_mult` and the logits by `output_mult /
            // mup_scale_factor`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            if flag(raw, "use_mup") {
                let (Some(input), Some(output), Some(scale)) = (
                    number(raw, "input_mult"),
                    number(raw, "output_mult"),
                    number(raw, "mup_scale_factor").filter(|scale| *scale != 0.0),
                ) else {
                    bail!(
                        "{} declares use_mup without input_mult, output_mult and a nonzero mup_scale_factor",
                        path.display()
                    );
                };
                architecture.embedding_multiplier = Some(input);
                architecture.logits_multiplier = Some(output / scale);
            }
        }
        "seed_oss" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = flag(raw, "attention_out_bias");
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
        }
        "solar" => {
            // Solar Pro: Llama's block with block skip connections — before
            // the layers `bskcn_1` and `bskcn_2` list the hidden state is
            // kept, and before those `bskcn_3` and `bskcn_4` list it is
            // blended with the first and second kept one by the inference
            // weight, the second of `bskcn_tv`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            fits(layers, path)?;
            let listed = |key: &str| -> Result<u128> {
                let Some(list) = raw.get(key).and_then(Value::as_array) else {
                    bail!("{} declares a Solar model without {key}", path.display());
                };
                list.iter().try_fold(0u128, |set, entry| {
                    match entry.as_u64().map(|layer| layer as usize).filter(|layer| *layer < layers) {
                        Some(layer) => Ok(set | (1u128 << layer)),
                        None => bail!("{} lists {entry} in {key}, which is not one of its {layers} layers", path.display()),
                    }
                })
            };
            let [_, inference] = numbers::<2>(raw, "bskcn_tv", path)?;
            architecture.skip_connections = Some(SkipConnections {
                save: [listed("bskcn_1")?, listed("bskcn_2")?],
                blend: [listed("bskcn_3")?, listed("bskcn_4")?],
                weight: inference,
            });
        }
        "PanguEmbedded" => {
            // openPangu-Embedded: Llama's block with `bias` on every
            // attention projection (`qkv_bias`, when stated, for query, key
            // and value), `mlp_bias`, and under `sandwich_norm` norms over
            // each sublayer's output.
            let bias = flag(raw, "attention_bias") || flag(raw, "bias");
            architecture.output_bias = bias;
            architecture.query_key_value_bias = raw.get("qkv_bias").and_then(Value::as_bool).unwrap_or(bias);
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            if flag(raw, "sandwich_norm") {
                architecture.output_norms = true;
                architecture.names = Names::PANGU_SANDWICH;
            }
        }
        "telechat3" => {
            // TeleChat3: Llama's block with `attention_bias` on every
            // attention projection and `mlp_bias`; its `telechat3-yarn`
            // rotation is read with the other scalings.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
        }
        "step1" => {
            // Step1 (Step-Audio's language model): Llama's block with
            // `num_attention_groups` key-value heads and no rotation;
            // positions enter as `-slope · sqrt(distance)` on the scores,
            // with ALiBi's slopes, as its `build_alibi_cache` builds them.
            architecture.positions = Positions::AlibiRoot;
        }
        "iquestcoder" => {
            // IQuest-Coder: Llama with `mlp_bias`, OLMo's `clip_qkv`, and
            // Qwen2's sliding window on the layers from `max_window_layers`
            // under `use_sliding_window`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.clip_qkv = number(raw, "clip_qkv");
            if flag(raw, "use_sliding_window") {
                architecture.sliding_window = whole(raw, "sliding_window");
                let from = whole(raw, "max_window_layers").unwrap_or(0);
                architecture.sliding_layers = every_layer(layers, path)? & !every_layer(from.min(layers), path)?;
            }
        }
        "arcee" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.names = Names::UP_DOWN;
        }
        "jais2" => {
            // Jais 2: Arcee's plain `up_proj`/`down_proj` feed-forward under
            // LayerNorms with bias; `attention_bias` and `mlp_bias` default
            // to true and `hidden_act` to squared ReLU, as Transformers'
            // `Jais2Config` defines them.
            let stated = |key: &str| raw.get(key).and_then(Value::as_bool).unwrap_or(true);
            architecture.query_key_value_bias = stated("attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = stated("mlp_bias");
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.names = Names::UP_DOWN;
            if text(raw, "hidden_act").is_none() {
                architecture.activation = Activation::Relu2;
            }
        }
        "ernie4_5" | "ernie4_5_moe" => {
            let bias = flag(raw, "use_bias");
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.interleaved_rotary = true;
            if model_type == "ernie4_5_moe" {
                // ERNIE 4.5's experts cover the layers from
                // `moe_layer_start_index` to `moe_layer_end_index` (the last
                // when negative) every `moe_layer_interval`; scores are a
                // softmax, chosen with `moe_statics.e_score_correction_bias`
                // added and renormalised, beside `moe_num_shared_experts`
                // shared experts.
                fits(layers, path)?;
                let start = whole(raw, "moe_layer_start_index").unwrap_or(0);
                let end = raw
                    .get("moe_layer_end_index")
                    .and_then(Value::as_i64)
                    .filter(|end| *end >= 0)
                    .map_or(layers.saturating_sub(1), |end| end as usize);
                let interval = whole(raw, "moe_layer_interval").unwrap_or(1).max(1);
                let dense = (0..layers)
                    .filter(|layer| *layer < start || *layer > end || (layer - start) % interval != 0)
                    .fold(0u128, |set, layer| set | (1u128 << layer));
                let mut routed = experts(
                    raw,
                    "moe_num_experts",
                    "moe_intermediate_size",
                    true,
                    ExpertLayout::Qwen,
                    dense,
                    path,
                )?;
                routed.shared = whole(raw, "moe_num_shared_experts")
                    .filter(|shared| *shared > 0)
                    .map(|shared| SharedExpert {
                        intermediate: shared * routed.intermediate,
                        module: "mlp.shared_experts",
                        gated: false,
                        form: SharedForm::GateUpDown,
                    });
                routed.selection_bias = Some("mlp.moe_statics.e_score_correction_bias");
                architecture.experts = Some(routed);
            }
        }
        "minicpm" => {
            // MiniCPM's muP scales: the embedding by `scale_emb`, each
            // sublayer by `scale_depth / sqrt(layers)`, and the logits by
            // `dim_model_base / hidden_size`.
            architecture.embedding_multiplier = number(raw, "scale_emb");
            architecture.residual_multiplier =
                number(raw, "scale_depth").map(|depth| depth / (layers as f64).sqrt());
            architecture.logits_multiplier = number(raw, "dim_model_base")
                .map(|base| base / llama.hidden_size as f64);
        }
        "orion" => {
            architecture.norm = NormKind::Layer { bias: true };
        }
        "mamba" | "falcon_mamba" => {
            // Mamba: every block is a norm and a selective state-space
            // mixer; Falcon-Mamba adds a scale-free RMS norm on the step and
            // the input and output matrices.
            let hidden = llama.hidden_size;
            let inner = whole(raw, "intermediate_size")
                .or_else(|| whole(raw, "expand").map(|expand| expand * hidden));
            let (Some(inner), Some(state), Some(kernel)) =
                (inner, whole(raw, "state_size"), whole(raw, "conv_kernel"))
            else {
                bail!(
                    "{} declares a state-space model without intermediate_size (or expand), state_size and conv_kernel",
                    path.display()
                );
            };
            let step_rank = match raw.get("time_step_rank") {
                Some(Value::String(rule)) if rule == "auto" => hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK),
                _ => whole(raw, "time_step_rank").unwrap_or(hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK)),
            };
            architecture.names = Names::MAMBA;
            architecture.positions = Positions::None;
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank,
                projection_bias: flag(raw, "use_bias"),
                convolution_bias: raw.get("use_conv_bias").and_then(Value::as_bool).unwrap_or(true),
                parameter_norm: if model_type == "falcon_mamba" {
                    ParameterNorm::Bare(number(raw, "mixer_rms_eps").unwrap_or(llama.rms_norm_eps))
                } else {
                    ParameterNorm::None
                },
                layers: every_layer(layers, path)?,
                feed_forward: false,
                structured: None,
            });
        }
        "jamba" => {
            // Jamba: attention on every `attn_layer_period`-th layer from
            // `attn_layer_offset` — with no position signal at all — and
            // Mamba elsewhere with weighted norms on its step and matrices;
            // every layer has a feed-forward, routed over `num_experts` on
            // every `expert_layer_period`-th layer from `expert_layer_offset`.
            fits(layers, path)?;
            let periodic = |period: &str, offset: &str| -> u128 {
                let period = whole(raw, period).unwrap_or(1).max(1);
                let offset = whole(raw, offset).unwrap_or(0);
                (0..layers)
                    .filter(|layer| layer % period == offset)
                    .fold(0, |set, layer| set | (1u128 << layer))
            };
            let attention = periodic("attn_layer_period", "attn_layer_offset");
            let routed = periodic("expert_layer_period", "expert_layer_offset");
            let hidden = llama.hidden_size;
            let inner = whole(raw, "mamba_expand").map(|expand| expand * hidden);
            let (Some(inner), Some(state), Some(kernel)) =
                (inner, whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
            else {
                bail!(
                    "{} declares a Jamba model without mamba_expand, mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            let step_rank = match raw.get("mamba_dt_rank") {
                Some(Value::String(rule)) if rule == "auto" => hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK),
                _ => whole(raw, "mamba_dt_rank").unwrap_or(hidden.div_ceil(MAMBA_WIDTH_PER_STEP_RANK)),
            };
            architecture.names = Names::JAMBA;
            architecture.positions = Positions::None;
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank,
                projection_bias: flag(raw, "mamba_proj_bias"),
                convolution_bias: raw.get("mamba_conv_bias").and_then(Value::as_bool).unwrap_or(true),
                parameter_norm: ParameterNorm::Weighted(llama.rms_norm_eps),
                layers: every_layer(layers, path)? & !attention,
                feed_forward: true,
                structured: None,
            });
            if whole(raw, "num_experts").is_some_and(|count| count > 1) {
                architecture.experts = Some(experts(
                    raw,
                    "num_experts",
                    "intermediate_size",
                    false,
                    ExpertLayout::Jamba,
                    every_layer(layers, path)? & !routed,
                    path,
                )?);
            }
        }
        "mamba2" => {
            // Mamba-2: every block is a norm and the structured mixer.
            let heads = structured(raw, "num_heads", "head_dim", "n_groups", path)?;
            let inner = heads.heads * heads.head_dim;
            let (Some(state), Some(kernel)) = (whole(raw, "state_size"), whole(raw, "conv_kernel"))
            else {
                bail!(
                    "{} declares a Mamba-2 model without state_size and conv_kernel",
                    path.display()
                );
            };
            architecture.names = Names::MAMBA;
            architecture.positions = Positions::None;
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "use_bias"),
                convolution_bias: raw.get("use_conv_bias").and_then(Value::as_bool).unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                layers: every_layer(layers, path)?,
                feed_forward: false,
                structured: Some(heads),
            });
        }
        "bamba" => {
            // Bamba: Mamba-2 mixers with attention on the layers
            // `attn_layer_indices` lists, a feed-forward after every mixer,
            // and rotation over `attn_rotary_emb` components of each head.
            fits(layers, path)?;
            let attention = raw
                .get("attn_layer_indices")
                .and_then(Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(Value::as_u64)
                        .filter(|layer| (*layer as usize) < layers)
                        .fold(0u128, |set, layer| set | (1u128 << layer))
                })
                .unwrap_or(0);
            let heads = structured(raw, "mamba_n_heads", "mamba_d_head", "mamba_n_groups", path)?;
            let (Some(state), Some(kernel)) = (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
            else {
                bail!(
                    "{} declares a Bamba model without mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            if let Some(width) = whole(raw, "attn_rotary_emb") {
                architecture.rotary_dim = width.min(architecture.head_dim);
            }
            architecture.names = Names::JAMBA;
            architecture.state_space = Some(StateSpaceSpec {
                inner: heads.heads * heads.head_dim,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "mamba_proj_bias"),
                convolution_bias: raw.get("mamba_conv_bias").and_then(Value::as_bool).unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                layers: every_layer(layers, path)? & !attention,
                feed_forward: true,
                structured: Some(heads),
            });
        }
        "nemotron_h" | "nemotron_h_puzzle" => {
            // Nemotron-H: one norm and one sublayer per layer, its kind the
            // layer's letter in `hybrid_override_pattern` — `M` a Mamba-2
            // scan, `*` attention without rotation, `-` a squared-ReLU
            // `up_proj`/`down_proj` feed-forward, `E` a mixture of such
            // feed-forwards behind DeepSeek-V3's router.
            fits(layers, path)?;
            let Some(pattern) = text(raw, "hybrid_override_pattern") else {
                bail!("{} declares a Nemotron-H model without hybrid_override_pattern", path.display());
            };
            if pattern.chars().count() != layers {
                bail!(
                    "{} spells {} layers in hybrid_override_pattern for {layers} layers",
                    path.display(),
                    pattern.chars().count()
                );
            }
            let (mut mamba, mut feed_forward, mut routed_layers) = (0u128, 0u128, 0u128);
            for (layer, kind) in pattern.chars().enumerate() {
                match kind {
                    'M' => mamba |= 1u128 << layer,
                    '-' => feed_forward |= 1u128 << layer,
                    'E' => routed_layers |= 1u128 << layer,
                    '*' => {}
                    other => bail!(
                        "{} marks layer {layer} {other:?} in hybrid_override_pattern; Ster implements Nemotron-H's M (Mamba-2), * (attention), - (feed-forward) and E (mixture-of-experts) layers",
                        path.display()
                    ),
                }
            }
            if routed_layers != 0 {
                // `E` layers: `n_routed_experts` experts under `mixer.gate`
                // (sigmoid scores, `mixer.gate.e_score_correction_bias` added
                // to choose, `n_group`/`topk_group` limits, renormalised under
                // `norm_topk_prob`, scaled by `routed_scaling_factor`) beside
                // `n_shared_experts` shared ones under `mixer.shared_experts`,
                // `moe_shared_expert_intermediate_size` wide each; every
                // projection is `up_proj`/`down_proj`. Under `moe_latent_size`
                // the routed experts work on `mixer.fc1_latent_proj(x)` and
                // return through `mixer.fc2_latent_proj`. Nemotron Puzzle's
                // `block_configs` give each `moe` layer its own
                // `num_experts_per_tok` and `moe_intermediate_size`.
                let mut routed = deepseek_experts(raw, model_type, layers, path)?;
                routed.layout = ExpertLayout::NemotronH;
                routed.dense_layers = every_layer(layers, path)? & !routed_layers;
                routed.selection_bias = Some("mixer.gate.e_score_correction_bias");
                routed.latent = whole(raw, "moe_latent_size").filter(|width| *width > 0);
                let shared_count = whole(raw, "n_shared_experts").unwrap_or(1).max(1);
                routed.shared = whole(raw, "moe_shared_expert_intermediate_size")
                    .filter(|width| *width > 0)
                    .map(|intermediate| SharedExpert {
                        intermediate: shared_count * intermediate,
                        module: "mixer.shared_experts",
                        gated: false,
                        form: SharedForm::UpDown,
                    });
                if let Some(blocks) = raw.get("block_configs").and_then(Value::as_array) {
                    if blocks.len() != layers {
                        bail!("{} lists {} block_configs for {layers} layers", path.display(), blocks.len());
                    }
                    architecture.expert_overrides = Some(
                        blocks
                            .iter()
                            .map(|block| {
                                let moe = block.get("block_type").and_then(Value::as_str) == Some("moe");
                                let top_k = whole(block, "num_experts_per_tok").unwrap_or(routed.top_k);
                                let width = whole(block, "moe_intermediate_size").unwrap_or(routed.intermediate);
                                if moe && (top_k == 0 || top_k > routed.count) {
                                    bail!(
                                        "{} routes a block's tokens to {top_k} of {} experts; it must be at least one and at most all of them",
                                        path.display(),
                                        routed.count
                                    );
                                }
                                Ok(moe.then_some((top_k, width)))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                architecture.experts = Some(routed);
            }
            let heads = structured(raw, "mamba_num_heads", "mamba_head_dim", "n_groups", path)?;
            let (Some(state), Some(kernel)) = (whole(raw, "ssm_state_size"), whole(raw, "conv_kernel"))
            else {
                bail!(
                    "{} declares a Nemotron-H model without ssm_state_size and conv_kernel",
                    path.display()
                );
            };
            architecture.names =
                if model_type == "nemotron_h_puzzle" { Names::NEMOTRON_H_MODEL } else { Names::NEMOTRON_H };
            architecture.positions = Positions::None;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.lone_sublayers = Some(feed_forward | routed_layers);
            architecture.state_space = Some(StateSpaceSpec {
                inner: heads.heads * heads.head_dim,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "use_bias"),
                convolution_bias: raw.get("use_conv_bias").and_then(Value::as_bool).unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                layers: mamba,
                feed_forward: false,
                structured: Some(heads),
            });
        }
        // Param2-MoE and Sarvam-30B (`sarvam_moe`) are Ling 2.0's block and
        // router under their own names; Ling 2.5 and 3.0 (`bailing_hybrid`)
        // add lightning attention.
        "bailing_moe" | "param2moe" | "bailing_hybrid" | "sarvam_moe" => {
            // Ling 1.x and 2.0: Llama's block under Bailing's names, one
            // stacked `query_key_value`, per-head query and key norms when
            // `use_qk_norm` holds, the first `first_k_dense_replace` layers
            // dense and the rest routed over `num_experts` experts
            // (`mlp.gate`, `mlp.experts.{e}`) beside
            // `moe_intermediate_size · num_shared_experts` shared ones. Ling
            // 1.x scores by softmax and renormalises under `norm_topk_prob`;
            // Ling 2.0's `score_function` sigmoid adds `mlp.gate.expert_bias`
            // to choose (`moe_router_enable_expert_bias`) within its best
            // `topk_group` of `n_group` groups, always renormalises and
            // scales by `routed_scaling_factor`.
            if flag(raw, "norm_head") || flag(raw, "norm_softmax") {
                bail!(
                    "{} normalises its output head (norm_head) or its softmax (norm_softmax); Ster implements Ling's plain lm_head",
                    path.display()
                );
            }
            architecture.names = Names::BAILING;
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.query_key_value_bias = flag(raw, "use_qkv_bias");
            architecture.output_bias = flag(raw, "use_bias");
            if model_type == "bailing_hybrid" {
                bailing_hybrid(raw, layers, llama, &mut architecture, path)?;
            } else if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
            }
            fits(layers, path)?;
            let first_dense = whole(raw, "first_k_dense_replace").unwrap_or(0).min(layers);
            let dense = (0..first_dense).fold(0u128, |set, layer| set | (1u128 << layer));
            let sigmoid = match text(raw, "score_function") {
                None | Some("softmax") => false,
                Some("sigmoid") => true,
                Some(other) => bail!(
                    "{} declares score_function {other:?}; Ster implements softmax and sigmoid expert scores",
                    path.display()
                ),
            };
            let mut routed =
                experts(raw, "num_experts", "moe_intermediate_size", false, ExpertLayout::Qwen, dense, path)?;
            routed.normalize = (sigmoid || flag(raw, "norm_topk_prob")) && routed.top_k > 1;
            routed.shared = whole(raw, "num_shared_experts")
                .filter(|shared| *shared > 0)
                .map(|shared| SharedExpert {
                    intermediate: shared * routed.intermediate,
                    module: "mlp.shared_experts",
                    gated: false,
                    form: SharedForm::GateUpDown,
                });
            if sigmoid {
                routed.scoring = Scoring::Sigmoid;
                routed.routed_scale = number(raw, "routed_scaling_factor");
                routed.selection_bias =
                    flag(raw, "moe_router_enable_expert_bias").then_some("mlp.gate.expert_bias");
                routed.groups = match (whole(raw, "n_group"), whole(raw, "topk_group")) {
                    (Some(groups), Some(chosen_groups)) => {
                        if groups == 0 || routed.count % groups != 0 || chosen_groups > groups {
                            bail!(
                                "{} splits {} experts into {groups} groups and keeps {chosen_groups}; the groups must divide the experts evenly and at least as many must exist as are kept",
                                path.display(),
                                routed.count
                            );
                        }
                        Some(ExpertGroups { groups, chosen_groups, rank_by_top_two: true })
                    }
                    _ => None,
                };
            }
            if model_type == "bailing_hybrid" {
                step_limits(raw, ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"), layers, &mut routed, path)?;
            }
            architecture.experts = Some(routed);
        }
        "falcon_h1" => {
            // Falcon-H1: every layer runs attention and a Mamba-2 scan side
            // by side on one normed input (`input_layernorm`), then
            // `pre_ff_layernorm` and the feed-forward, all under Jamba's
            // names, with muP multipliers on the embeddings, the attention
            // input and output, every key, the scan input, the five parts
            // of its projection and its output, the feed-forward's gate and
            // output, and the logits.
            let scale = |key: &str| number(raw, key).unwrap_or(1.0);
            if raw.get("attn_layer_indices").is_some_and(|indices| !indices.is_null())
                || raw.get("mamba_use_mlp").and_then(Value::as_bool) == Some(false)
                || flag(raw, "mamba_norm_before_gate")
                || flag(raw, "mamba_proj_bias") != flag(raw, "projectors_bias")
            {
                bail!(
                    "{} declares attn_layer_indices, mamba_use_mlp false, mamba_norm_before_gate, or a projectors_bias unlike mamba_proj_bias; Ster implements Falcon-H1 with attention, a scan and a feed-forward on every layer, the gate before the norm, and one bias setting for the scan's projections",
                    path.display()
                );
            }
            let mut heads = structured(raw, "mamba_n_heads", "mamba_d_head", "mamba_n_groups", path)?;
            let (Some(state), Some(kernel)) = (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
            else {
                bail!(
                    "{} declares a Falcon-H1 model without mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            let inner = whole(raw, "mamba_d_ssm").unwrap_or_else(|| {
                (number(raw, "mamba_expand").unwrap_or(2.0) * llama.hidden_size as f64) as usize
            });
            if inner != heads.heads * heads.head_dim {
                bail!(
                    "{} sizes its scan at {inner} but {} heads of {} make {}; the inner width must be the heads' total",
                    path.display(),
                    heads.heads,
                    heads.head_dim,
                    heads.heads * heads.head_dim
                );
            }
            let [gate_scale, output_scale] = numbers::<2>(raw, "mlp_multipliers", path)?;
            heads.input_scale = scale("ssm_in_multiplier");
            heads.projection_scales = Some(numbers::<5>(raw, "ssm_multipliers", path)?);
            heads.gated_norm = raw.get("mamba_rms_norm").and_then(Value::as_bool).unwrap_or(true);
            architecture.names = Names::JAMBA;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            architecture.embedding_multiplier = number(raw, "embedding_multiplier");
            architecture.logits_multiplier = number(raw, "lm_head_multiplier");
            architecture.key_scale = number(raw, "key_multiplier");
            architecture.feed_forward_scales = Some((gate_scale, output_scale));
            architecture.parallel_scan = Some(ParallelScan {
                attention_in: scale("attention_in_multiplier"),
                attention_out: scale("attention_out_multiplier"),
                scan_out: scale("ssm_out_multiplier"),
            });
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "mamba_proj_bias"),
                convolution_bias: raw.get("mamba_conv_bias").and_then(Value::as_bool).unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                // The scan runs inside every layer's parallel block, never
                // as a layer of its own.
                layers: 0,
                feed_forward: true,
                structured: Some(heads),
            });
        }
        "glm4_moe" => {
            // GLM-4.5's MoE: biased query, key and value, optional per-head
            // query and key norms, rotation over `partial_rotary_factor` by
            // halves, and DeepSeek-V3's router and shared experts.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
            }
            architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
        }
        "internlm2" => {
            // InternLM2: Llama's block under its own names, query, key and
            // value fused in `wqkv` by key-value group, and `bias` on every
            // attention projection.
            architecture.names = Names::INTERNLM2;
            architecture.qkv_layout = QkvLayout::Grouped;
            architecture.query_key_value_bias = flag(raw, "bias");
            architecture.output_bias = architecture.query_key_value_bias;
        }
        "lfm2" | "lfm2_moe" => {
            // LFM2: gated short convolutions on every layer but those
            // `full_attn_idxs` (or `layer_types`' `full_attention`) names,
            // per-head query and key norms on the attention layers.
            fits(layers, path)?;
            let attention: u128 = match raw.get("layer_types").and_then(Value::as_array) {
                Some(types) => types
                    .iter()
                    .enumerate()
                    .filter(|(_, kind)| kind.as_str() == Some("full_attention"))
                    .fold(0, |set, (layer, _)| set | (1u128 << layer)),
                None => raw
                    .get("full_attn_idxs")
                    .and_then(Value::as_array)
                    .map(|list| {
                        list.iter()
                            .filter_map(Value::as_u64)
                            .filter(|layer| (*layer as usize) < layers)
                            .fold(0u128, |set, layer| set | (1u128 << layer))
                    })
                    .unwrap_or(0),
            };
            let Some(kernel) = whole(raw, "conv_L_cache") else {
                bail!("{} declares an LFM2 model without conv_L_cache", path.display());
            };
            architecture.names = Names::LFM2;
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.short_convolution = Some(ShortConvolution {
                kernel,
                bias: flag(raw, "conv_bias"),
                layers: every_layer(layers, path)? & !attention,
            });
            if model_type == "lfm2_moe" {
                // LFM2-MoE: the first `num_dense_layers` layers keep the
                // plain `intermediate_size` feed-forward; the rest route
                // over `num_experts` experts (`feed_forward.gate`,
                // `feed_forward.experts.{e}.w1`/`w3`/`w2`) by sigmoid
                // scores, chosen with `feed_forward.expert_bias` added when
                // `use_expert_bias` holds, renormalised under
                // `norm_topk_prob`, scaled by `routed_scaling_factor`.
                let dense_count = whole(raw, "num_dense_layers").unwrap_or(0).min(layers);
                let dense = (0..dense_count).fold(0u128, |set, layer| set | (1u128 << layer));
                let mut routed = experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    flag(raw, "norm_topk_prob"),
                    ExpertLayout::Lfm2,
                    dense,
                    path,
                )?;
                routed.scoring = Scoring::Sigmoid;
                routed.selection_bias = flag(raw, "use_expert_bias").then_some("feed_forward.expert_bias");
                routed.routed_scale = number(raw, "routed_scaling_factor");
                architecture.experts = Some(routed);
            }
        }
        "gpt_oss" => {
            // GPT-OSS: biased attention projections with a learned sink per
            // head, sliding-window layers from `layer_types`, and a mixture
            // of `num_local_experts` biased experts behind a biased router,
            // the top `num_experts_per_tok` renormalised, each with the
            // clamped gate `swiglu_limit` bounds.
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.attention_sinks = every_layer(layers, path)?;
            architecture.sliding_window = whole(raw, "sliding_window");
            let mut routed = experts(
                raw,
                "num_local_experts",
                "intermediate_size",
                true,
                ExpertLayout::GptOss,
                0,
                path,
            )?;
            routed.swiglu_limit = number(raw, "swiglu_limit").map(SwigluLimit::GptOss);
            architecture.experts = Some(routed);
        }
        "hunyuan_v1_dense" | "hunyuan_v1_moe" => {
            // HunYuan: per-head query and key norms named `query_layernorm`
            // and `key_layernorm`, applied after the rotation.
            architecture.names = Names::HUNYUAN;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.feed_forward_bias = flag(raw, "mlp_bias");
            if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.norm_after_rotary = true;
            }
            if model_type == "hunyuan_v1_moe" {
                if flag(raw, "use_cla") || flag(raw, "use_mla") {
                    bail!(
                        "{} shares attention across layers (use_cla) or compresses it (use_mla); Ster implements HunYuan's own attention on every layer",
                        path.display()
                    );
                }
                // HunYuan-MoE: `num_experts` experts under `mlp.gate.wg`,
                // the top `moe_topk` weighted by their renormalised softmax,
                // beside `mlp.shared_mlp`, `num_shared_expert` times
                // `intermediate_size` wide, unless `use_mixed_mlp_moe` is
                // false; the first `moe_layer_num_skipped` layers stay dense.
                let width = if raw.get("moe_intermediate_size").is_some() {
                    "moe_intermediate_size"
                } else {
                    "intermediate_size"
                };
                fits(layers, path)?;
                let skipped = whole(raw, "moe_layer_num_skipped").unwrap_or(0).min(layers);
                let dense = (0..skipped).fold(0u128, |set, layer| set | (1u128 << layer));
                let mut routed =
                    experts(raw, "num_experts", width, true, ExpertLayout::HunYuan, dense, path)?;
                if raw.get("use_mixed_mlp_moe").and_then(Value::as_bool) != Some(false) {
                    let shared = uniform(raw, "num_shared_expert", path)?.unwrap_or(1);
                    routed.shared = (shared > 0).then_some(SharedExpert {
                        intermediate: shared * llama.intermediate_size,
                        module: "mlp.shared_mlp",
                        gated: false,
                        form: SharedForm::GateUpDown,
                    });
                }
                architecture.experts = Some(routed);
            }
        }
        "exaone" => {
            // EXAONE 3 and 3.5: Llama's block under GPT-2-style names.
            architecture.names = Names::EXAONE;
        }
        "telechat" => {
            // TeleChat2: Llama's block below `transformer`, a separate
            // `query` beside a `key_value` matrix paired per head, biases on
            // the attention output and the down projection only.
            if flag(raw, "apply_residual_connection_post_layernorm") || flag(raw, "embed_layernorm") {
                bail!(
                    "{} adds its residual after the norm (apply_residual_connection_post_layernorm) or normalises the embeddings (embed_layernorm); Ster implements TeleChat2's pre-norm block without an embedding norm",
                    path.display()
                );
            }
            architecture.names = Names::TELECHAT;
            architecture.qkv_layout = QkvLayout::PairedKeyValue;
            architecture.output_bias = true;
            architecture.down_bias = true;
        }
        // GLM-4.7-Flash (`glm4_moe_lite`) is DeepSeek-V3's latent attention
        // and router under GLM's name.
        // DeepSeek-V3.2 and GLM-5 (`glm_moe_dsa`) add DeepSeek Sparse
        // Attention's indexer.
        // openPangu-Ultra-MoE is DeepSeek-V3's latent attention and a sigmoid
        // router without selection bias, under its own key names (read as
        // DeepSeek's), with sandwich norms.
        "deepseek_v2" | "deepseek_v3" | "minicpm3" | "glm4_moe_lite" | "deepseek_v32" | "glm_moe_dsa" | "axk1" | "pangu_ultra_moe" => {
            if architecture.latent.is_none() {
                bail!(
                    "{} declares no kv_lora_rank; Ster implements {model_type} with its latent attention",
                    path.display()
                );
            }
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            if model_type == "minicpm3" {
                // MiniCPM's muP scales; its rotation is by halves.
                architecture.embedding_multiplier = number(raw, "scale_emb");
                architecture.residual_multiplier =
                    number(raw, "scale_depth").map(|depth| depth / (layers as f64).sqrt());
                architecture.logits_multiplier = number(raw, "dim_model_base")
                    .map(|base| base / llama.hidden_size as f64);
            } else {
                // DeepSeek rotates adjacent pairs (`rope_interleave`, on by
                // default).
                architecture.interleaved_rotary =
                    raw.get("rope_interleave").and_then(Value::as_bool).unwrap_or(true);
            }
            if raw.get("n_routed_experts").is_some_and(|count| !count.is_null()) {
                architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
            }
            // Mistral Large 3 (read from Mistral's own format) adds Llama
            // 4's query temperature (`llama_4_scaling`).
            architecture.query_temperature = llama4_temperature(raw, scaling, layers, path)?;
            if matches!(model_type, "deepseek_v32" | "glm_moe_dsa") {
                architecture.sparse_index = Some(sparse_index(raw, model_type, layers, path)?);
            }
            // A.X-K1 normalises each routed layer's feed-forward output by
            // `post_mlp_layernorm` before the residual add.
            if model_type == "axk1" {
                architecture.names = Names::AXK1;
                architecture.routed_output_norm = true;
            }
            // openPangu-Ultra-MoE's `sandwich_norm`: norms over each
            // sublayer's output, `pre_mlp_layernorm` before the feed-forward.
            if flag(raw, "sandwich_norm") {
                architecture.output_norms = true;
                architecture.names = Names::PANGU_SANDWICH;
            }
        }
        "step3_text" => {
            // Step3 (its text decoder, `text_config` of `step3_vl`): the
            // query narrows through `q_proj` to `share_q_dim`, passes the RMS
            // norm `inter_norm` and widens through `wq`, beside
            // `num_attention_groups` key-value heads, and Step's experts.
            let Some(width) = whole(raw, "share_q_dim").filter(|width| *width > 0) else {
                bail!("{} declares a Step3 model without share_q_dim", path.display());
            };
            architecture.names = Names::STEP3;
            architecture.query_bottleneck = Some(width);
            architecture.experts = Some(step_experts(raw, layers, path)?);
        }
        "step3p5" => step3p5(raw, layers, llama, &mut architecture, path)?,
        "k2_horizon" => k2_horizon(raw, layers, llama, &mut architecture, path)?,
        "longcat_flash" => longcat_flash(raw, llama, &mut architecture, path)?,
        "longcat_flash_ngram" => {
            longcat_flash(raw, llama, &mut architecture, path)?;
            architecture.ngram = Some(longcat_ngram(raw, path)?);
        }
        "sarvam_mla" => sarvam_mla(raw, layers, &mut architecture, path)?,
        "cohere2_moe" => cohere2_moe(raw, layers, &mut architecture, path)?,
        "llama4_text" => llama4(raw, layers, &mut architecture, path)?,
        "afmoe" => afmoe(raw, layers, llama, &mut architecture, path)?,
        "laguna" => laguna(raw, scaling, layers, llama, &mut architecture, path)?,
        "muse_glimmer_text" => muse_glimmer(raw, layers, llama, &mut architecture, path)?,
        "deepseek" => {
            // DeepSeek-MoE (v1): Llama's attention, rotating by halves, and
            // DeepSeek's experts — `n_routed_experts` scored by softmax and
            // `n_shared_experts` shared ones, dense before
            // `first_k_dense_replace` and off `moe_layer_freq`.
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
        }
        "zamba2" => {
            // Zamba2: Mamba-2 layers, and on the layers `layers_block_type`
            // calls `hybrid` a shared transformer block (one of
            // `num_mem_blocks`, used in turn) whose output joins the Mamba-2
            // mixer's input. The block's attention reads the hidden state
            // beside the embeddings, `attention_hidden_size` wide, divides
            // its scores by `sqrt(attention_head_dim / 2)` and rotates only
            // under `use_mem_rope`.
            fits(layers, path)?;
            let Some(kinds) = raw.get("layers_block_type").and_then(Value::as_array) else {
                bail!("{} declares a Zamba2 model without layers_block_type", path.display());
            };
            if kinds.len() != layers {
                bail!("{} lists {} layers_block_type entries for {layers} layers", path.display(), kinds.len());
            }
            let mut hybrid = 0u128;
            for (layer, kind) in kinds.iter().enumerate() {
                match kind.as_str() {
                    Some("hybrid") => hybrid |= 1u128 << layer,
                    Some("mamba") => {}
                    other => bail!(
                        "{} names layer {layer} {other:?}; a Zamba2 layer is mamba or hybrid",
                        path.display()
                    ),
                }
            }
            let mut heads = structured(raw, "n_mamba_heads", "mamba_headdim", "mamba_ngroups", path)?;
            // Transformers' `Zamba2MambaMixer` clamps the step to
            // `(time_step_min, inf)`.
            heads.step_limit = (number(raw, "time_step_min").unwrap_or(0.0), f64::INFINITY);
            let (Some(state), Some(kernel)) = (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv"))
            else {
                bail!(
                    "{} declares a Zamba2 model without mamba_d_state and mamba_d_conv",
                    path.display()
                );
            };
            let inner = (number(raw, "mamba_expand").unwrap_or(2.0) * llama.hidden_size as f64) as usize;
            if inner != heads.heads * heads.head_dim {
                bail!(
                    "{} sizes its Mamba-2 mixer at mamba_expand times the width, {inner}, but {} heads of {} make {}",
                    path.display(),
                    heads.heads,
                    heads.head_dim,
                    heads.heads * heads.head_dim
                );
            }
            let Some(blocks) = whole(raw, "num_mem_blocks").filter(|blocks| *blocks > 0) else {
                bail!("{} declares a Zamba2 model without num_mem_blocks", path.display());
            };
            let attention_adapters = flag(raw, "use_shared_attention_adapter");
            let feed_forward_adapters = flag(raw, "use_shared_mlp_adapter");
            let rank = whole(raw, "adapter_rank").unwrap_or(0);
            if (attention_adapters || feed_forward_adapters) && rank == 0 {
                bail!(
                    "{} turns on Zamba2's shared adapters without an adapter_rank",
                    path.display()
                );
            }
            architecture.names = Names::JAMBA;
            architecture.score_divisor = (architecture.head_dim as f64 / 2.0).sqrt();
            if !flag(raw, "use_mem_rope") {
                architecture.positions = Positions::None;
            }
            architecture.shared_blocks = Some(SharedBlocksSpec {
                hybrid_layers: hybrid,
                blocks,
                attention_input: whole(raw, "attention_hidden_size").unwrap_or(2 * llama.hidden_size),
                intermediate: llama.intermediate_size,
                rank,
                attention_adapters,
                feed_forward_adapters,
            });
            architecture.state_space = Some(StateSpaceSpec {
                inner,
                state,
                kernel,
                step_rank: 0,
                projection_bias: flag(raw, "add_bias_linear"),
                convolution_bias: raw.get("use_conv_bias").and_then(Value::as_bool).unwrap_or(true),
                parameter_norm: ParameterNorm::None,
                layers: every_layer(layers, path)? & !hybrid,
                feed_forward: false,
                structured: Some(heads),
            });
        }
        "minimax_text_01" | "minimax" => {
            // MiniMax-Text-01 (its own release spells `minimax_text_01`,
            // Transformers `minimax`): lightning attention on the layers
            // `attn_type_list` marks 0 (or `layer_types` calls
            // `linear_attention`), full attention over `rotary_dim` of each
            // head on the rest, Mixtral-named experts after every mixer, and
            // every sublayer joining the residual as `residual · alpha +
            // output · beta`, the residual being the normed input under
            // `postnorm`.
            fits(layers, path)?;
            let mut lightning = 0u128;
            if let Some(kinds) = raw.get("attn_type_list").and_then(Value::as_array) {
                if kinds.len() != layers {
                    bail!("{} lists {} attn_type_list entries for {layers} layers", path.display(), kinds.len());
                }
                for (layer, kind) in kinds.iter().enumerate() {
                    match kind.as_u64() {
                        Some(0) => lightning |= 1u128 << layer,
                        Some(1) => {}
                        _ => bail!(
                            "{} marks layer {layer} {kind} in attn_type_list; MiniMax layers are 0 (lightning) or 1 (full attention)",
                            path.display()
                        ),
                    }
                }
            } else if let Some(types) = raw.get("layer_types").and_then(Value::as_array) {
                if types.len() != layers {
                    bail!("{} lists {} layer_types for {layers} layers", path.display(), types.len());
                }
                for (layer, kind) in types.iter().enumerate() {
                    match kind.as_str() {
                        Some("linear_attention") => lightning |= 1u128 << layer,
                        Some("full_attention") => {}
                        other => bail!(
                            "{} names layer {layer} {other:?}; a MiniMax layer is linear_attention or full_attention",
                            path.display()
                        ),
                    }
                }
            } else {
                bail!(
                    "{} declares a MiniMax model with neither attn_type_list nor layer_types",
                    path.display()
                );
            }
            if whole(raw, "shared_intermediate_size").unwrap_or(0) > 0 {
                bail!(
                    "{} declares shared experts (shared_intermediate_size); Ster implements MiniMax-Text-01 with routed experts only",
                    path.display()
                );
            }
            // The release's keys first, Transformers' after; one when unstated.
            let pair = |alpha: [&str; 2], beta: [&str; 2]| -> (f64, f64) {
                let read = |keys: [&str; 2]| keys.iter().find_map(|key| number(raw, key)).unwrap_or(1.0);
                (read(alpha), read(beta))
            };
            architecture.lightning = Some(LightningSpec {
                heads: llama.num_attention_heads,
                head_dim: architecture.head_dim,
                layers: lightning,
                form: LightningForm::MiniMax,
            });
            architecture.scaled_residuals = Some(ScaledResiduals {
                from_normed: raw.get("postnorm").and_then(Value::as_bool).unwrap_or(true),
                linear_attention: pair(
                    ["layernorm_linear_attention_alpha", "linear_attn_alpha_factor"],
                    ["layernorm_linear_attention_beta", "linear_attn_beta_factor"],
                ),
                full_attention: pair(
                    ["layernorm_full_attention_alpha", "full_attn_alpha_factor"],
                    ["layernorm_full_attention_beta", "full_attn_beta_factor"],
                ),
                feed_forward: pair(
                    ["layernorm_mlp_alpha", "mlp_alpha_factor"],
                    ["layernorm_mlp_beta", "mlp_beta_factor"],
                ),
            });
            architecture.experts = Some(experts(
                raw,
                "num_local_experts",
                "intermediate_size",
                true,
                ExpertLayout::Mixtral,
                0,
                path,
            )?);
        }
        "minimax_m2" => {
            // MiniMax-M2: full attention on every layer with query and key
            // norms over the whole projection (`qk_norm_type` per_layer) or
            // per head, rotation over `rotary_dim` of each head, and a
            // mixture of `num_local_experts` Mixtral-named experts scored by
            // sigmoid, chosen with `block_sparse_moe.e_score_correction_bias`
            // added (`use_routing_bias`) and renormalised.
            let linear_layers = raw
                .get("attn_type_list")
                .and_then(Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_u64() == Some(0)));
            if linear_layers || whole(raw, "shared_intermediate_size").unwrap_or(0) > 0 {
                bail!(
                    "{} declares lightning-attention layers (attn_type_list 0) or shared experts (shared_intermediate_size); Ster implements MiniMax-M2 with full attention everywhere and routed experts only",
                    path.display()
                );
            }
            if flag(raw, "use_qk_norm") {
                architecture.query_key_norm = match text(raw, "qk_norm_type") {
                    None | Some("per_layer") => QueryKeyNorm::Full,
                    Some("per_head") => QueryKeyNorm::PerHead,
                    Some(other) => bail!(
                        "{} declares qk_norm_type {other:?}; Ster implements per_layer and per_head",
                        path.display()
                    ),
                };
            }
            let mut routed = experts(
                raw,
                "num_local_experts",
                "intermediate_size",
                true,
                ExpertLayout::Mixtral,
                0,
                path,
            )?;
            routed.scoring = match text(raw, "scoring_func") {
                None | Some("sigmoid") => Scoring::Sigmoid,
                Some("softmax") => Scoring::Softmax,
                Some(other) => bail!(
                    "{} declares scoring_func {other:?}; Ster implements softmax and sigmoid expert scores",
                    path.display()
                ),
            };
            routed.selection_bias = (raw.get("use_routing_bias").and_then(Value::as_bool) != Some(false))
                .then_some("block_sparse_moe.e_score_correction_bias");
            architecture.experts = Some(routed);
        }
        "kimi_linear" => {
            // Kimi-Linear: Kimi Delta Attention on the layers
            // `linear_attn_config.kda_layers` names (counted from one),
            // multi-head latent attention on the rest (unrotated when
            // `mla_use_nope` holds), and a mixture of `num_experts`
            // Mixtral-named experts under `block_sparse_moe` beside shared
            // experts, routed as DeepSeek-V3 routes.
            fits(layers, path)?;
            if architecture.latent.is_none() {
                bail!(
                    "{} declares no kv_lora_rank; Ster implements Kimi-Linear's attention layers as latent attention",
                    path.display()
                );
            }
            let Some(linear) = raw.get("linear_attn_config").and_then(Value::as_object) else {
                bail!("{} declares a Kimi-Linear model without linear_attn_config", path.display());
            };
            let size = |key: &str| -> Result<usize> {
                linear
                    .get(key)
                    .and_then(Value::as_u64)
                    .map(|size| size as usize)
                    .filter(|size| *size > 0)
                    .with_context(|| format!("{} declares linear_attn_config without {key}", path.display()))
            };
            let Some(listed) = linear.get("kda_layers").and_then(Value::as_array) else {
                bail!("{} declares linear_attn_config without kda_layers", path.display());
            };
            let mut kda = 0u128;
            for entry in listed {
                match entry.as_u64().map(|layer| layer as usize) {
                    Some(layer) if (1..=layers).contains(&layer) => kda |= 1u128 << (layer - 1),
                    _ => bail!(
                        "{} lists kda_layers entry {entry}, outside layers 1 to {layers}",
                        path.display()
                    ),
                }
            }
            let (heads, head_dim) = (size("num_heads")?, size("head_dim")?);
            architecture.delta_rule = Some(DeltaRuleSpec {
                key_heads: heads,
                value_heads: heads,
                key_dim: head_dim,
                value_dim: head_dim,
                kernel: size("short_conv_kernel_size")?,
                layers: kda,
                form: DeltaRuleForm::Kimi,
                negative_eigenvalues: false,
                decay_floor: None,
            });
            if flag(raw, "mla_use_nope") {
                architecture.positions = Positions::None;
            }
            let first_dense = whole(raw, "first_k_dense_replace").unwrap_or(0);
            let frequency = whole(raw, "moe_layer_freq").unwrap_or(1).max(1);
            let dense = (0..layers)
                .filter(|layer| *layer < first_dense || layer % frequency != 0)
                .fold(0u128, |set, layer| set | (1u128 << layer));
            let mut routed = experts(
                raw,
                "num_experts",
                "moe_intermediate_size",
                flag(raw, "moe_renormalize"),
                ExpertLayout::Mixtral,
                dense,
                path,
            )?;
            routed.scoring = match text(raw, "moe_router_activation_func") {
                None | Some("softmax") => Scoring::Softmax,
                Some("sigmoid") => Scoring::Sigmoid,
                Some(other) => bail!(
                    "{} declares moe_router_activation_func {other:?}; Ster implements softmax and sigmoid expert scores",
                    path.display()
                ),
            };
            routed.selection_bias = Some("block_sparse_moe.gate.e_score_correction_bias");
            routed.routed_scale = number(raw, "routed_scaling_factor");
            if flag(raw, "use_grouped_topk") {
                let groups = whole(raw, "num_expert_group").unwrap_or(1);
                let chosen_groups = whole(raw, "topk_group").unwrap_or(groups);
                if groups == 0 || routed.count % groups != 0 || chosen_groups > groups {
                    bail!(
                        "{} splits {} experts into {groups} groups and keeps {chosen_groups}; the groups must divide the experts evenly and at least as many must exist as are kept",
                        path.display(),
                        routed.count
                    );
                }
                routed.groups = Some(ExpertGroups { groups, chosen_groups, rank_by_top_two: true });
            }
            routed.shared = whole(raw, "num_shared_experts")
                .filter(|shared| *shared > 0)
                .map(|shared| SharedExpert {
                    intermediate: shared * routed.intermediate,
                    module: "block_sparse_moe.shared_experts",
                    gated: false,
                    form: SharedForm::GateUpDown,
                });
            architecture.experts = Some(routed);
        }
        "gpt_neox" => {
            // Pythia and GPT-NeoX: LayerNorm with bias, a head-interleaved
            // `query_key_value`, biased projections and a plain GELU
            // feed-forward; parallel blocks by default, each half with its own
            // norm.
            architecture.names = Names::GPT_NEOX;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.qkv_layout = QkvLayout::Grouped;
            let bias = raw.get("attention_bias").and_then(Value::as_bool).unwrap_or(true);
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.parallel =
                raw.get("use_parallel_residual").and_then(Value::as_bool).unwrap_or(true);
            architecture.parallel_norms = true;
        }
        "gptj" => {
            // GPT-J: one `ln_1` before both halves of a parallel block,
            // unbiased attention, a biased plain feed-forward and head, and
            // interleaved rotation over `rotary_dim`.
            architecture.names = Names::GPT_J;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.parallel = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.lm_head_bias = true;
            architecture.interleaved_rotary = true;
        }
        "gpt2" | "gpt_bigcode" => {
            // GPT-2 and GPT-BigCode: a learned position table, LayerNorm with
            // bias, one biased `c_attn` holding query, key and value (one
            // key-value head when GPT-BigCode's `multi_query`), and a biased
            // plain feed-forward. GPT-2 stores every projection `[inputs,
            // outputs]`.
            if flag(raw, "scale_attn_by_inverse_layer_idx") {
                bail!(
                    "{} declares scale_attn_by_inverse_layer_idx; Ster scales every layer's attention by the head width alone",
                    path.display()
                );
            }
            architecture.names = Names::GPT2;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.positions = Positions::Learned { offset: 0 };
            architecture.conv1d = model_type == "gpt2";
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
        }
        "opt" => {
            // OPT: a learned position table read two rows in, pre-norm
            // blocks with biased projections and a plain ReLU feed-forward.
            if raw.get("do_layer_norm_before").and_then(Value::as_bool) == Some(false) {
                bail!(
                    "{} declares do_layer_norm_before false; Ster implements OPT's pre-norm blocks",
                    path.display()
                );
            }
            if let Some(width) = whole(raw, "word_embed_proj_dim").filter(|w| *w != llama.hidden_size) {
                bail!(
                    "{} projects {width}-wide word embeddings into a {}-wide model (word_embed_proj_dim); Ster implements OPT with embeddings as wide as the model",
                    path.display(),
                    llama.hidden_size
                );
            }
            let bias = raw.get("enable_bias").and_then(Value::as_bool).unwrap_or(true);
            architecture.names = Names::OPT;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.positions = Positions::Learned { offset: OPT_POSITION_OFFSET };
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
        }
        "bloom" => {
            // BLOOM: ALiBi instead of positions, a norm after the embedding,
            // a head-interleaved biased `query_key_value`, and a biased plain
            // feed-forward with the tanh GELU.
            if flag(raw, "apply_residual_connection_post_layernorm") {
                bail!(
                    "{} declares apply_residual_connection_post_layernorm; Ster implements BLOOM with the residual taken before each norm",
                    path.display()
                );
            }
            architecture.names = Names::BLOOM;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.embedding_norm = true;
            architecture.positions = Positions::Alibi { inside_scale: false };
            architecture.qkv_layout = QkvLayout::Grouped;
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.activation = Activation::GeluTanh;
        }
        "falcon" => {
            // Falcon: BLOOM's tensor names, rows grouped by key-value head
            // (one group for the multi-query models, `num_kv_heads` for the
            // new decoder architecture, one per head otherwise), a parallel
            // block with one norm or, in the new architecture, `ln_attn` and
            // `ln_mlp`, the exact GELU, and rotation or ALiBi.
            let new = flag(raw, "new_decoder_architecture");
            let two_norms = new && whole(raw, "num_ln_in_parallel_attn") != Some(1);
            architecture.names = if two_norms { Names::FALCON_TWO_NORMS } else { Names::FALCON };
            architecture.parallel = new || flag(raw, "parallel_attn");
            architecture.parallel_norms = two_norms;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.qkv_layout = QkvLayout::Grouped;
            let bias = flag(raw, "bias");
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.activation = Activation::Gelu;
            if flag(raw, "alibi") {
                architecture.positions = Positions::Alibi { inside_scale: true };
            }
        }
        "mpt" => {
            // MPT: `transformer.blocks`, LayerNorm and projections without
            // bias under `no_bias`, one `Wqkv`, the exact GELU, and ALiBi or a
            // learned `wpe` as `attn_config` says.
            let attention = raw.get("attn_config").cloned().unwrap_or(Value::Null);
            let unsupported = [
                ("qk_ln", flag(&attention, "qk_ln")),
                ("clip_qkv", attention.get("clip_qkv").is_some_and(|v| !v.is_null())),
                ("softmax_scale", attention.get("softmax_scale").is_some_and(|v| !v.is_null())),
                (
                    "attn_type",
                    text(&attention, "attn_type").is_some_and(|kind| kind != "multihead_attention"),
                ),
                (
                    "alibi_bias_max",
                    number(&attention, "alibi_bias_max").is_some_and(|max| max != ALIBI_SPAN),
                ),
            ];
            if let Some((key, _)) = unsupported.iter().find(|(_, set)| *set) {
                bail!(
                    "{} declares attn_config.{key} other than MPT's defaults; Ster implements MPT's plain multi-head attention with ALiBi's standard slopes",
                    path.display()
                );
            }
            let alibi = flag(&attention, "alibi");
            if alibi && !llama.num_attention_heads.is_power_of_two() {
                bail!(
                    "{} uses ALiBi over {} heads; Ster implements MPT's slopes for a power-of-two head count",
                    path.display(),
                    llama.num_attention_heads
                );
            }
            let bias = !raw.get("no_bias").and_then(Value::as_bool).unwrap_or(true);
            architecture.names = Names::MPT;
            architecture.norm = NormKind::Layer { bias };
            architecture.positions = if alibi {
                Positions::Alibi { inside_scale: false }
            } else {
                Positions::Learned { offset: 0 }
            };
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.activation = Activation::Gelu;
        }
        "chatglm" => {
            // ChatGLM (GLM-4-9B and ChatGLM2/3): Llama's block under its own
            // names, query, key and value stacked in `query_key_value`
            // (biased under `add_qkv_bias`), gate and up stacked in
            // `dense_h_to_4h`, the first half of each head rotating —
            // adjacent pairs under `original_rope` — and RMS norms under
            // `rmsnorm`, LayerNorms otherwise.
            if flag(raw, "apply_residual_connection_post_layernorm") {
                bail!(
                    "{} adds its residual after the norm (apply_residual_connection_post_layernorm); Ster implements ChatGLM's pre-norm block",
                    path.display()
                );
            }
            if raw.get("post_layer_norm").and_then(Value::as_bool) == Some(false) {
                bail!(
                    "{} declares no final norm (post_layer_norm false); Ster implements ChatGLM with its final_layernorm",
                    path.display()
                );
            }
            architecture.names = Names::CHATGLM;
            architecture.qkv_layout = QkvLayout::Stacked;
            architecture.fused_feed_forward = true;
            let bias = flag(raw, "add_bias_linear");
            architecture.query_key_value_bias = bias || flag(raw, "add_qkv_bias");
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.interleaved_rotary = raw.get("original_rope").and_then(Value::as_bool).unwrap_or(true);
            if raw.get("rmsnorm").and_then(Value::as_bool) == Some(false) {
                architecture.norm = NormKind::Layer { bias: true };
            }
        }
        "glm" | "glm4" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.fused_feed_forward = true;
            architecture.interleaved_rotary = true;
            if model_type == "glm4" {
                architecture.output_norms = true;
                architecture.names = Names::GLM4;
            }
        }
        // VaultGemma is Gemma 2 without the norms over each sublayer's
        // output.
        "gemma" | "gemma2" | "gemma3_text" | "vaultgemma" => {
            architecture.norm_offset = true;
            architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
            architecture.activation = Activation::GeluTanh;
            if model_type != "gemma" {
                architecture.output_norms = model_type != "vaultgemma";
                architecture.names = Names::GEMMA2;
                architecture.sliding_window = whole(raw, "sliding_window");
                if let Some(scalar) = number(raw, "query_pre_attn_scalar") {
                    architecture.score_divisor = scalar.sqrt();
                }
                architecture.attention_softcap = number(raw, "attn_logit_softcapping");
                architecture.final_softcap = number(raw, "final_logit_softcapping");
            }
            if model_type == "gemma2" || model_type == "vaultgemma" {
                architecture.sliding_layers = even_layers(layers, path)?;
            }
            if model_type == "gemma3_text" {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.local_rope_theta =
                    number(raw, "rope_local_base_freq").map(|theta| theta as f32);
                // Rnj-1.5 lists `chunked_attention` layers: chunks of
                // `sliding_window` positions, each query seeing its own
                // chunk and the one before it.
                let chunked = raw
                    .get("layer_types")
                    .and_then(Value::as_array)
                    .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("chunked_attention")));
                if chunked {
                    architecture.chunk_lookback = Some(RNJ1_CHUNK_LOOKBACK);
                }
                // Older Gemma 3 configs state the pattern instead of
                // `layer_types`: every `sliding_window_pattern`-th layer is
                // global, the rest local.
                if let Some(pattern) = whole(raw, "sliding_window_pattern").filter(|p| *p > 0) {
                    fits(layers, path)?;
                    architecture.sliding_layers = (0..layers)
                        .filter(|layer| (layer + 1) % pattern != 0)
                        .fold(0, |set, layer| set | (1u128 << layer));
                }
            }
        }
        "gemma4_text" | "gemma4_unified_text" => {
            // Gemma 4: Gemma 3's norms around both sublayers and per-head
            // query and key norms, but norms that scale by their weight
            // itself, scores left undivided, a scale-free norm over each
            // head's value, and every block's output times its
            // `layer_scalar`. Its full-attention layers have wider heads
            // (`global_head_dim`), under `attention_k_eq_v` their own
            // key-value heads (`num_global_key_value_heads`) and values taken
            // from the key projection, and rotate by the `full_attention`
            // rotation over that width; the sliding-window layers rotate by
            // their own base.
            if raw.get("use_bidirectional_attention").and_then(Value::as_str) == Some("all") {
                bail!(
                    "{} declares use_bidirectional_attention \"all\", so every token sees the whole sequence; Ster runs causal decoders only",
                    path.display()
                );
            }
            if flag(raw, "use_double_wide_mlp") {
                bail!(
                    "{} declares use_double_wide_mlp, a doubled feed-forward on the key-value-sharing layers; Ster builds every Gemma 4 feed-forward intermediate_size wide",
                    path.display()
                );
            }
            let Some(global_head_dim) = whole(raw, "global_head_dim") else {
                bail!("{} declares a Gemma 4 model without global_head_dim", path.display());
            };
            let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
                bail!("{} declares a Gemma 4 model without layer_types", path.display());
            };
            architecture.local_rope_theta = number(raw, "rope_local_base_freq").map(|theta| theta as f32);
            if architecture.local_rope_theta.is_none() {
                bail!(
                    "{} declares a Gemma 4 model without a sliding_attention rotation base",
                    path.display()
                );
            }
            architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
            architecture.activation = Activation::GeluTanh;
            architecture.output_norms = true;
            architecture.names = Names::GEMMA2;
            architecture.query_key_norm = QueryKeyNorm::PerHead;
            architecture.value_norm = true;
            architecture.score_divisor = 1.0;
            architecture.layer_scalar = true;
            architecture.final_softcap = number(raw, "final_logit_softcapping");
            architecture.sliding_window = whole(raw, "sliding_window");
            architecture.sliding_layers = listed_layers(types, layers, path)?;
            let key_is_value = flag(raw, "attention_k_eq_v");
            architecture.global_attention = Some(GlobalAttention {
                heads: None,
                head_dim: global_head_dim,
                // Transformers gives the full-attention layers their own
                // key-value head count only under `attention_k_eq_v`.
                key_value_heads: whole(raw, "num_global_key_value_heads").filter(|_| key_is_value),
                rotary_dim: global_head_dim,
                key_is_value,
            });
            architecture.rope_scaling = rope_scaling(scaling, global_head_dim, raw, llama, path)?;
            // The last `num_kv_shared_layers` layers reuse the keys and
            // values of the last earlier layer of their kind.
            let shared = whole(raw, "num_kv_shared_layers").unwrap_or(0);
            if shared > 0 {
                if shared >= layers {
                    bail!(
                        "{} shares keys and values on {shared} of {layers} layers, leaving none to produce them",
                        path.display()
                    );
                }
                let first = layers - shared;
                architecture.shared_key_values = Some(first);
                if let Some(layer) = (first..layers).find(|layer| architecture.key_value_source(*layer).is_none()) {
                    bail!(
                        "{} shares keys and values from layer {first} on, but layer {layer} has no earlier layer of its kind to share them from",
                        path.display()
                    );
                }
            }
            if let Some(width) = whole(raw, "hidden_size_per_layer_input").filter(|width| *width > 0) {
                architecture.per_layer_input = Some(PerLayerInputSpec {
                    width,
                    vocab: whole(raw, "vocab_size_per_layer_input").unwrap_or(llama.vocab_size),
                });
            }
            if flag(raw, "enable_moe_block") {
                architecture.side_experts = Some(experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    true,
                    ExpertLayout::Gemma4,
                    0,
                    path,
                )?);
            }
        }
        "mimo_v2_flash" | "mimo_v2" => mimo_v2(raw, model_type, layers, llama, &mut architecture, path)?,
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    // A rotation stated per layer kind (Transformers 5's `rope_parameters`
    // by layer type) gives sliding-window layers their own base, whatever
    // the family, so a scaling on the full-attention rotation (Mellum's
    // YaRN) never reaches them.
    if architecture.local_rope_theta.is_none() && architecture.sliding_window.is_some() {
        architecture.local_rope_theta = number(raw, "rope_local_base_freq").map(|theta| theta as f32);
    }
    // A family with recurrent mixers (LFM2's convolutions, Qwen3-Next's,
    // Kimi-Linear's and OLMo Hybrid's delta rule, MiniMax's lightning
    // attention, Granite 4.0's Mamba-2) reads its `layer_types` as which
    // layers run which mixer, above; every other family's say which layers
    // attend through the window.
    let mixers_listed = architecture.short_convolution.is_some()
        || architecture.delta_rule.is_some()
        || architecture.lightning.is_some()
        || architecture.state_space.is_some();
    let windows = raw.get("layer_types").and_then(Value::as_array).filter(|_| !mixers_listed);
    if let Some(types) = windows {
        architecture.sliding_layers = windowed_layers(types, layers, architecture.chunk_lookback.is_some(), path)?;
    }
    // Cohere 2's global layers apply no rotary embedding; so do EXAONE 4's
    // when the model mixes local and global layers at all.
    let hybrid_exaone = matches!(model_type, "exaone4" | "exaone_moe") && architecture.sliding_window.is_some();
    if model_type == "cohere2" || hybrid_exaone {
        architecture.unrotated_layers = every_layer(layers, path)? & !architecture.sliding_layers;
    }
    if architecture.sliding_layers != 0 && architecture.sliding_window.is_none() {
        bail!(
            "{} makes some layers sliding-window but declares no sliding_window",
            path.display()
        );
    }
    Ok(architecture)
}

/// The mixture of experts a config declares: how many under `count_key`, how
/// many per token under `num_experts_per_tok`, each one's width under
/// `width_key`, each stated once or once per layer.
fn experts(
    raw: &Value,
    count_key: &str,
    width_key: &str,
    normalize: bool,
    layout: ExpertLayout,
    dense_layers: u128,
    path: &Path,
) -> Result<MixtureOfExperts> {
    let (Some(count), Some(top_k), Some(intermediate)) = (
        uniform(raw, count_key, path)?,
        uniform(raw, "num_experts_per_tok", path)?,
        uniform(raw, width_key, path)?,
    ) else {
        bail!(
            "{} declares a mixture of experts without {count_key}, num_experts_per_tok and {width_key}",
            path.display()
        );
    };
    if top_k == 0 || top_k > count {
        bail!(
            "{} routes each token to {top_k} of {count} experts; it must be at least one and at most all of them",
            path.display()
        );
    }
    Ok(MixtureOfExperts {
        count,
        top_k,
        intermediate,
        normalize,
        shared: None,
        layout,
        dense_layers,
        scoring: Scoring::Softmax,
        groups: None,
        selection_bias: None,
        routed_scale: None,
        swiglu_limit: None,
        identity_experts: 0,
        average_shared: false,
        weight_input: false,
        latent: None,
    })
}

/// A whole number a config states once, or once per layer as HunYuan's
/// lists do. A per-layer list must hold one value throughout, because Ster
/// builds every mixture-of-experts layer alike.
fn uniform(raw: &Value, key: &str, path: &Path) -> Result<Option<usize>> {
    let Some(Value::Array(values)) = raw.get(key) else {
        return Ok(whole(raw, key));
    };
    let values: Option<Vec<u64>> = values.iter().map(Value::as_u64).collect();
    match values.as_deref() {
        Some([first, rest @ ..]) if rest.iter().all(|value| value == first) => {
            Ok(Some(*first as usize))
        }
        _ => bail!(
            "{} lists {key} values that differ between layers or are not whole numbers; Ster builds every mixture-of-experts layer alike",
            path.display()
        ),
    }
}

/// DeepSeek's mixture of experts: `n_routed_experts` routed and
/// `n_shared_experts` shared experts of `moe_intermediate_size` each, dense
/// layers before `first_k_dense_replace` and off `moe_layer_freq`, scores by
/// `scoring_func`, chosen by `topk_method` (`greedy`, or group-limited over
/// `n_group` groups keeping `topk_group`; `noaux_tc` ranks groups by their
/// best two and adds `e_score_correction_bias` to choose), and weights scaled
/// by `routed_scaling_factor` — always on V3, on V2 only when they are not
/// renormalised.
fn deepseek_experts(
    raw: &Value,
    model_type: &str,
    layers: usize,
    path: &Path,
) -> Result<MixtureOfExperts> {
    fits(layers, path)?;
    let first_dense = whole(raw, "first_k_dense_replace").unwrap_or(0);
    let frequency = whole(raw, "moe_layer_freq").unwrap_or(1).max(1);
    let dense_layers = if raw.get("mlp_layer_types").is_some_and(Value::is_array) {
        qwen_dense_layers(raw, layers, path)?
    } else if raw.get("moe_layer_freq").is_some_and(Value::is_array) {
        // MiMo-V2 lists `moe_layer_freq` per layer: one for routed, zero
        // for dense.
        every_layer(layers, path)? & !flagged_layers(raw, "moe_layer_freq", layers, path)?
    } else {
        (0..layers)
            .filter(|layer| *layer < first_dense || layer % frequency != 0)
            .fold(0, |set, layer| set | (1u128 << layer))
    };
    let normalize = flag(raw, "norm_topk_prob");
    let mut routed = experts(
        raw,
        // K-EXAONE spells the counts `num_experts` and `num_shared_experts`.
        if model_type == "exaone_moe" { "num_experts" } else { "n_routed_experts" },
        "moe_intermediate_size",
        normalize,
        ExpertLayout::Qwen,
        dense_layers,
        path,
    )?;
    routed.shared = whole(raw, if model_type == "exaone_moe" { "num_shared_experts" } else { "n_shared_experts" })
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    // GLM-4-MoE's and Nemotron-H's routers are DeepSeek-V3's and their
    // configs leave the method out: sigmoid scores, `noaux_tc` selection.
    let v3_default =
        matches!(model_type, "glm4_moe" | "glm4_moe_lite" | "glm_moe_dsa" | "exaone_moe") || model_type.starts_with("nemotron_h");
    let v3_router = v3_default || matches!(model_type, "deepseek_v3" | "deepseek_v32" | "axk1" | "pangu_ultra_moe");
    routed.scoring = match text(raw, "scoring_func") {
        None if v3_default => Scoring::Sigmoid,
        None | Some("softmax") => Scoring::Softmax,
        Some("sigmoid") => Scoring::Sigmoid,
        Some(other) => bail!(
            "{} declares scoring_func {other:?}; Ster implements softmax and sigmoid expert scores",
            path.display()
        ),
    };
    let method = text(raw, "topk_method").unwrap_or(if v3_default { "noaux_tc" } else { "greedy" });
    // A.X-K1's `topk_method` `none` is vLLM's grouped top-k with no
    // selection bias, which ranks each group by its best expert: DeepSeek-V2's
    // `group_limited_greedy`.
    let method = if model_type == "axk1" && method == "none" { "group_limited_greedy" } else { method };
    routed.groups = match method {
        "greedy" => None,
        "group_limited_greedy" | "noaux_tc" => {
            let (Some(groups), Some(chosen_groups)) = (whole(raw, "n_group"), whole(raw, "topk_group"))
            else {
                bail!(
                    "{} routes by {method} without n_group and topk_group",
                    path.display()
                );
            };
            if groups == 0 || routed.count % groups != 0 || chosen_groups > groups {
                bail!(
                    "{} splits {} experts into {groups} groups and keeps {chosen_groups}; the groups must divide the experts evenly and at least as many must exist as are kept",
                    path.display(),
                    routed.count
                );
            }
            Some(ExpertGroups {
                groups,
                chosen_groups,
                rank_by_top_two: method == "noaux_tc",
            })
        }
        other => bail!(
            "{} declares topk_method {other:?}; Ster implements greedy, group_limited_greedy and noaux_tc",
            path.display()
        ),
    };
    // K-EXAONE keeps the selection bias beside the router rather than in it.
    let bias = if model_type == "exaone_moe" {
        "mlp.e_score_correction_bias"
    } else {
        "mlp.gate.e_score_correction_bias"
    };
    routed.selection_bias = (method == "noaux_tc").then_some(bias);
    let scale = number(raw, "routed_scaling_factor");
    routed.routed_scale = if v3_router || !(normalize && routed.top_k > 1) {
        scale
    } else {
        None
    };
    Ok(routed)
}

/// Granite 4.0 (`granitemoehybrid`): Mamba-2 mixers (`mamba`, read from
/// Bamba's `mamba_*` keys) on the layers `layer_types` calls `mamba`,
/// attention on the others, rotation only when `position_embedding_type` is
/// `rope`, and after every mixer either GraniteMoE's experts beside the
/// stacked `shared_mlp` or, with no `num_local_experts`, the `shared_mlp`
/// alone as the dense feed-forward.
fn granite_hybrid(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!("{} declares a Granite 4.0 model without layer_types", path.display());
    };
    if types.len() != layers {
        bail!("{} lists {} layer_types for {layers} layers", path.display(), types.len());
    }
    let mut mamba = 0u128;
    for (layer, kind) in types.iter().enumerate() {
        match kind.as_str() {
            Some("mamba") => mamba |= 1u128 << layer,
            Some("attention") => {}
            other => bail!(
                "{} names layer {layer} {other:?}; a Granite 4.0 layer is mamba or attention",
                path.display()
            ),
        }
    }
    architecture.positions = match text(raw, "position_embedding_type") {
        None | Some("rope") => Positions::Rotary,
        Some("nope") => Positions::None,
        Some(other) => bail!(
            "{} declares position_embedding_type {other:?}; Granite 4.0 rotates (rope) or does not (nope)",
            path.display()
        ),
    };
    let heads = structured(raw, "mamba_n_heads", "mamba_d_head", "mamba_n_groups", path)?;
    let (Some(state), Some(kernel)) = (whole(raw, "mamba_d_state"), whole(raw, "mamba_d_conv")) else {
        bail!(
            "{} declares a Granite 4.0 model without mamba_d_state and mamba_d_conv",
            path.display()
        );
    };
    architecture.state_space = Some(StateSpaceSpec {
        inner: heads.heads * heads.head_dim,
        state,
        kernel,
        step_rank: 0,
        projection_bias: flag(raw, "mamba_proj_bias"),
        convolution_bias: raw.get("mamba_conv_bias").and_then(Value::as_bool).unwrap_or(true),
        parameter_norm: ParameterNorm::None,
        layers: mamba,
        feed_forward: true,
        structured: Some(heads),
    });
    let Some(shared) = whole(raw, "shared_intermediate_size") else {
        bail!(
            "{} declares a Granite 4.0 model without shared_intermediate_size",
            path.display()
        );
    };
    if whole(raw, "num_local_experts").unwrap_or(0) == 0 {
        architecture.names = Names::GRANITE_HYBRID;
        architecture.fused_feed_forward = true;
    } else {
        let mut routed =
            experts(raw, "num_local_experts", "intermediate_size", true, ExpertLayout::Granite, 0, path)?;
        routed.shared = Some(SharedExpert {
            intermediate: shared,
            module: "shared_mlp",
            gated: false,
            form: SharedForm::Stacked,
        });
        architecture.experts = Some(routed);
    }
    Ok(())
}

/// Mamba-2's head layout from the keys a family spells it with: the head
/// count, the head width and the group count (one when unstated), and
/// `time_step_limit` (unbounded when unstated or stated as non-finite).
fn structured(
    raw: &Value,
    heads_key: &str,
    width_key: &str,
    groups_key: &str,
    path: &Path,
) -> Result<StructuredSpec> {
    let (Some(heads), Some(head_dim)) = (whole(raw, heads_key), whole(raw, width_key)) else {
        bail!(
            "{} declares a Mamba-2 mixer without {heads_key} and {width_key}",
            path.display()
        );
    };
    let groups = whole(raw, groups_key).unwrap_or(1);
    if groups == 0 || heads % groups != 0 {
        bail!(
            "{} splits {heads} Mamba-2 heads into {groups} groups; the groups must divide the heads evenly",
            path.display()
        );
    }
    let limit = raw.get("time_step_limit").and_then(Value::as_array);
    let bound = |index: usize, unstated: f64| {
        limit
            .and_then(|limit| limit.get(index))
            .and_then(Value::as_f64)
            .unwrap_or(unstated)
    };
    Ok(StructuredSpec {
        heads,
        head_dim,
        groups,
        step_limit: (bound(0, 0.0), bound(1, f64::INFINITY)),
        input_scale: 1.0,
        projection_scales: None,
        gated_norm: true,
    })
}

/// Qwen MoE's dense layers: those `mlp_layer_types` calls `dense` when the
/// config lists them (Mellum), otherwise those in `mlp_only_layers` and
/// those whose position is not a multiple of `decoder_sparse_step`.
fn qwen_dense_layers(raw: &Value, layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
    if let Some(kinds) = raw.get("mlp_layer_types").and_then(Value::as_array) {
        if kinds.len() != layers {
            bail!("{} lists {} mlp_layer_types for {layers} layers", path.display(), kinds.len());
        }
        let mut dense = 0u128;
        for (layer, kind) in kinds.iter().enumerate() {
            match kind.as_str() {
                Some("dense") => dense |= 1u128 << layer,
                Some("sparse") => {}
                other => bail!(
                    "{} names layer {layer}'s feed-forward {other:?}; mlp_layer_types entries are dense or sparse",
                    path.display()
                ),
            }
        }
        return Ok(dense);
    }
    let step = whole(raw, "decoder_sparse_step").unwrap_or(1).max(1);
    let listed: Vec<usize> = raw
        .get("mlp_only_layers")
        .and_then(Value::as_array)
        .map(|list| list.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect())
        .unwrap_or_default();
    Ok((0..layers)
        .filter(|layer| listed.contains(layer) || (layer + 1) % step != 0)
        .fold(0, |set, layer| set | (1u128 << layer)))
}

/// How often GraniteSWA's default layout attends fully: every fourth layer
/// from the first (`GraniteSWAConfig.__post_init__`, `i % 4 == 0`).
const GRANITE_SWA_FULL_EVERY: usize = 4;

/// How often OLMo Hybrid's default layout attends: each fourth layer from
/// the fourth (`OlmoHybridConfig.__post_init__`, `i % 4 == 3`).
const OLMO_HYBRID_FULL_EVERY: usize = 4;

/// IQuest-LoopCoder's defaults when its config leaves them out: two passes
/// and a 64-position local window (vLLM's `iquest_loopcoder.py`,
/// `getattr(config, "loop_num", 2)` and `getattr(config,
/// "loop_window_size", 64)`).
const LOOP_CODER_LOOPS: usize = 2;
const LOOP_CODER_WINDOW: usize = 64;

/// DeepSeek Sparse Attention's defaults when a config leaves them out
/// (Transformers' `DeepseekV32Config`: `index_topk` 2048, `index_head_dim`
/// 128, `index_n_heads` 64; `GlmMoeDsaConfig` narrows the heads to 32;
/// `index_topk_freq` 1 and `index_skip_topk_offset` 2).
const INDEX_TOP_K: usize = 2048;
const INDEX_HEAD_DIM: usize = 128;
const DEEPSEEK_INDEX_HEADS: usize = 64;
const GLM_INDEX_HEADS: usize = 32;
const INDEX_FREQUENCY: usize = 1;
const INDEX_SKIP_OFFSET: usize = 2;

/// DeepSeek Sparse Attention's indexer from the config. GLM-5 marks each
/// layer `full` or `shared` in `indexer_types` (or as `F` and `S` in
/// `index_topk_pattern`), or derives it from `index_topk_freq` and
/// `index_skip_topk_offset` as `GlmMoeDsaConfig.__post_init__` does: layer
/// `i` indexes when `max(i − offset + 1, 0)` is a multiple of the frequency.
fn sparse_index(raw: &Value, model_type: &str, layers: usize, path: &Path) -> Result<IndexerSpec> {
    fits(layers, path)?;
    let glm = model_type == "glm_moe_dsa";
    let mut shared_layers = 0u128;
    if glm {
        let listed: Option<Vec<bool>> = match (raw.get("indexer_types"), raw.get("index_topk_pattern")) {
            (Some(Value::Array(kinds)), _) => Some(
                kinds
                    .iter()
                    .enumerate()
                    .map(|(layer, kind)| match kind.as_str() {
                        Some("full") => Ok(false),
                        Some("shared") => Ok(true),
                        other => bail!(
                            "{} names layer {layer}'s indexer {other:?}; an indexer is full or shared",
                            path.display()
                        ),
                    })
                    .collect::<Result<_>>()?,
            ),
            (_, Some(Value::String(pattern))) => Some(
                pattern
                    .chars()
                    .enumerate()
                    .map(|(layer, mark)| match mark {
                        'F' => Ok(false),
                        'S' => Ok(true),
                        other => bail!(
                            "{} marks layer {layer}'s indexer {other:?} in index_topk_pattern; a mark is F or S",
                            path.display()
                        ),
                    })
                    .collect::<Result<_>>()?,
            ),
            _ => None,
        };
        let shared: Vec<bool> = match listed {
            Some(shared) => shared,
            None => {
                let frequency = whole(raw, "index_topk_freq").unwrap_or(INDEX_FREQUENCY).max(1);
                let offset = whole(raw, "index_skip_topk_offset").unwrap_or(INDEX_SKIP_OFFSET);
                (0..layers).map(|layer| (layer + 1).saturating_sub(offset) % frequency != 0).collect()
            }
        };
        if shared.len() != layers {
            bail!("{} marks {} layers' indexers for {layers} layers", path.display(), shared.len());
        }
        if shared.first() == Some(&true) {
            bail!("{} makes layer 0's indexer shared, with no earlier layer to share from", path.display());
        }
        shared_layers = shared
            .iter()
            .enumerate()
            .filter(|(_, shared)| **shared)
            .fold(0u128, |set, (layer, _)| set | (1u128 << layer));
    }
    Ok(IndexerSpec {
        heads: whole(raw, "index_n_heads").unwrap_or(if glm { GLM_INDEX_HEADS } else { DEEPSEEK_INDEX_HEADS }),
        head_dim: whole(raw, "index_head_dim").unwrap_or(INDEX_HEAD_DIM),
        top_k: whole(raw, "index_topk").unwrap_or(INDEX_TOP_K),
        interleaved: glm && raw.get("indexer_rope_interleave").and_then(Value::as_bool).unwrap_or(true),
        shared_layers,
    })
}

/// Ling 2.5's and 3.0's additions to Ling 2.0: DeepSeek's latent attention
/// (rotating adjacent pairs under `rope_interleave`) on every
/// `layer_group_size`-th layer, and linear attention on the others. Ling
/// 2.5's is lightning attention, its heads `head_dim` wide with their first
/// `partial_rotary_factor` share rotated by halves at the latent attention's
/// base; Ling 3.0's (`BailingMoeV3ForCausalLM`, or any config stating
/// `short_conv_kernel_size`) is Kimi Delta Attention, see [`ling_kda`].
fn bailing_hybrid(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    let Some(latent) = architecture.latent else {
        bail!("{} declares a Ling hybrid model without kv_lora_rank", path.display());
    };
    let ling3 = raw.get("short_conv_kernel_size").is_some()
        || raw
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|names| names.iter().any(|name| name.as_str() == Some("BailingMoeV3ForCausalLM")));
    if ling3 {
        return ling_kda(raw, layers, llama, architecture, path);
    }
    let group = whole(raw, "layer_group_size").unwrap_or(1).max(1);
    let linear = (0..layers)
        .filter(|layer| (layer + 1) % group != 0)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let Some(head_dim) = whole(raw, "head_dim") else {
        bail!("{} declares a Ling hybrid model without head_dim", path.display());
    };
    let heads = llama.num_attention_heads;
    if whole(raw, "num_kv_heads_for_linear_attn").is_some_and(|stated| stated != heads) {
        bail!(
            "{} gives its lightning attention a key-value head count other than its {heads} heads; Ster runs it with one key and value per head",
            path.display()
        );
    }
    let share = number(raw, "partial_rotary_factor").unwrap_or(1.0);
    let rotated = (head_dim as f64 * share) as usize;
    if rotated != latent.rotated {
        bail!(
            "{} rotates {rotated} components of each lightning head and {} of each latent one; Ster rotates both by one table",
            path.display(),
            latent.rotated
        );
    }
    let groups = whole(raw, "group_norm_size").unwrap_or(1).max(1);
    if (heads * head_dim) % groups != 0 {
        bail!("{} splits its {} lightning channels into {groups} norm groups unevenly", path.display(), heads * head_dim);
    }
    architecture.interleaved_rotary = raw.get("rope_interleave").and_then(Value::as_bool).unwrap_or(true);
    architecture.lightning = Some(LightningSpec {
        heads,
        head_dim,
        layers: linear,
        form: LightningForm::Bailing {
            groups,
            silu: flag(raw, "linear_silu") || flag(raw, "use_linear_silu"),
            qk_norm: flag(raw, "use_qk_norm"),
        },
    });
    Ok(())
}

/// Ling 3.0's linear layers: Kimi Delta Attention under `attention`, with
/// full-rank `f_proj` (the decay input, one per key channel) and `g_proj`
/// (the output gate) where Kimi factors them, `num_attention_heads` heads of
/// `head_dim` for query, key and value, and `short_conv_kernel_size` taps.
/// The decay is `kda_lower_bound · sigmoid(exp(A_log) · (f + dt_bias))`
/// under `kda_safe_gate` (vLLM's `bailing_moe_v3.py` and FLA's
/// `naive_kda_lowerbound_gate`), Kimi's `-exp(A_log) · softplus(f +
/// dt_bias)` otherwise. A layer is linear unless it closes a
/// `layer_group_size` group or follows the last whole group
/// (`_is_kda_layer`). The latent-attention layers multiply each head's
/// output by the sigmoid of its `attention.g_proj` logit under
/// `gated_attention_proj_granularity_type` `head_wise`.
fn ling_kda(raw: &Value, layers: usize, llama: &LlamaConfig, architecture: &mut Architecture, path: &Path) -> Result<()> {
    if raw.get("no_kda_lora").and_then(Value::as_bool) == Some(false) || flag(raw, "use_kda_lora") {
        bail!(
            "{} factors its Kimi Delta Attention projections (no_kda_lora false or use_kda_lora); Ster implements Ling 3.0's full-rank f_proj and g_proj",
            path.display()
        );
    }
    for key in ["use_nGPT", "value_norm", "up_proj_norm", "use_mla_nope"] {
        if flag(raw, key) {
            bail!("{} declares {key}; Ster implements Ling 3.0 without it", path.display());
        }
    }
    let Some(head_dim) = whole(raw, "head_dim") else {
        bail!("{} declares a Ling 3.0 model without head_dim", path.display());
    };
    let Some(kernel) = whole(raw, "short_conv_kernel_size").filter(|kernel| *kernel > 0) else {
        bail!("{} declares a Ling 3.0 model without short_conv_kernel_size", path.display());
    };
    let group = whole(raw, "layer_group_size").unwrap_or(1).max(1);
    let whole_groups = layers / group * group;
    let linear = (0..layers)
        .filter(|layer| (layer + 1) % group != 0 && *layer < whole_groups)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let heads = llama.num_attention_heads;
    let decay_floor = if raw.get("kda_safe_gate").and_then(Value::as_bool).unwrap_or(true) {
        Some(number(raw, "kda_lower_bound").unwrap_or(KDA_LOWER_BOUND))
    } else {
        None
    };
    architecture.delta_rule = Some(DeltaRuleSpec {
        key_heads: heads,
        value_heads: heads,
        key_dim: head_dim,
        value_dim: head_dim,
        kernel,
        layers: linear,
        form: DeltaRuleForm::Ling,
        negative_eigenvalues: false,
        decay_floor,
    });
    architecture.interleaved_rotary = raw.get("rope_interleave").and_then(Value::as_bool).unwrap_or(true);
    architecture.head_gate = match text(raw, "gated_attention_proj_granularity_type") {
        None => None,
        Some("head_wise") => Some(GateFunction::Sigmoid),
        Some(other) => bail!(
            "{} declares gated_attention_proj_granularity_type {other:?}; Ster implements Ling 3.0's head_wise gate",
            path.display()
        ),
    };
    Ok(())
}

/// The floor Kimi Delta Attention's safe gate puts under each log-decay when
/// a config leaves `kda_lower_bound` out (vLLM's `bailing_moe_v3.py`,
/// `getattr(config, "kda_lower_bound", -5.0)`).
const KDA_LOWER_BOUND: f64 = -5.0;

/// MiMo-V2 (`mimo_v2_flash`, `mimo_v2`): sliding-window layers where
/// `hybrid_layer_pattern` is one, with `swa_num_key_value_heads` key-value
/// heads (Ster's base count; the full-attention layers'
/// `num_key_value_heads` moves to `num_global_key_value_heads` as the config
/// is read), a learned sink per head under `add_swa_attention_sink_bias`
/// and their own base `swa_rope_theta`; full-attention layers carry sinks
/// under `add_full_attention_sink_bias`. Every head's query and key are
/// `head_dim` wide with the first `partial_rotary_factor` share rotated by
/// halves, its value `v_head_dim` wide and multiplied by
/// `attention_value_scale`. Query, key and value are separate, or stacked
/// in `qkv_proj` under `attention_projection_layout` `fused_qkv`. Layers
/// `moe_layer_freq` marks run DeepSeek-V3's routed experts.
fn mimo_v2(
    raw: &Value,
    model_type: &str,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let pairs = [
        ("num_attention_heads", "swa_num_attention_heads"),
        ("head_dim", "swa_head_dim"),
        ("v_head_dim", "swa_v_head_dim"),
    ];
    for (full, windowed) in pairs {
        if let (Some(stated), Some(own)) = (whole(raw, full), whole(raw, windowed)) {
            if stated != own {
                bail!(
                    "{} gives its sliding-window layers {windowed} {own} and its full-attention layers {full} {stated}; Ster builds both kinds with one head count and width",
                    path.display()
                );
            }
        }
    }
    architecture.names = Names::MIMO_V2;
    architecture.qkv_layout = match text(raw, "attention_projection_layout") {
        None => QkvLayout::Separate,
        Some("fused_qkv") => QkvLayout::Stacked,
        Some(other) => bail!(
            "{} declares attention_projection_layout {other:?}; Ster implements separate projections and fused_qkv",
            path.display()
        ),
    };
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    let windowed = flagged_layers(raw, "hybrid_layer_pattern", layers, path)?;
    architecture.sliding_layers = windowed;
    architecture.sliding_window = whole(raw, "sliding_window_size").or_else(|| whole(raw, "sliding_window"));
    let full = every_layer(layers, path)? & !windowed;
    architecture.attention_sinks = if flag(raw, "add_swa_attention_sink_bias") { windowed } else { 0 }
        | if flag(raw, "add_full_attention_sink_bias") { full } else { 0 };
    architecture.local_rope_theta = number(raw, "swa_rope_theta")
        .filter(|base| *base != f64::from(llama.rope_theta))
        .map(|base| base as f32);
    architecture.value_head_dim = whole(raw, "v_head_dim");
    architecture.value_scale = number(raw, "attention_value_scale");
    architecture.global_attention = Some(GlobalAttention {
        heads: None,
        head_dim: architecture.head_dim,
        key_value_heads: whole(raw, "num_global_key_value_heads"),
        rotary_dim: architecture.rotary_dim,
        key_is_value: false,
    });
    architecture.experts = Some(deepseek_experts(raw, model_type, layers, path)?);
    Ok(())
}

/// Step's experts (Step3, Step 3.5): the layers `moe_layers_enum` lists
/// (from zero; all but the first when absent) route over `moe_num_experts`
/// experts stacked under `moe`, renormalised under `norm_expert_weight`,
/// beside a shared `share_expert` `share_expert_dim` wide.
fn step_experts(raw: &Value, layers: usize, path: &Path) -> Result<MixtureOfExperts> {
    fits(layers, path)?;
    let routed_layers = match text(raw, "moe_layers_enum") {
        Some(list) => {
            let mut set = 0u128;
            for entry in list.split(',') {
                match entry.trim().parse::<usize>() {
                    Ok(layer) if layer < layers => set |= 1u128 << layer,
                    _ => bail!(
                        "{} lists moe_layers_enum entry {entry:?}, which names no layer below {layers}",
                        path.display()
                    ),
                }
            }
            set
        }
        None => every_layer(layers, path)? & !1u128,
    };
    let mut routed = experts(
        raw,
        "moe_num_experts",
        "moe_intermediate_size",
        flag(raw, "norm_expert_weight"),
        ExpertLayout::Step3,
        every_layer(layers, path)? & !routed_layers,
        path,
    )?;
    routed.shared = whole(raw, "share_expert_dim")
        .filter(|width| *width > 0)
        .map(|intermediate| SharedExpert {
            intermediate,
            module: "share_expert",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    Ok(routed)
}

/// The share of each ChatGLM head that rotates (vLLM's `chatglm.py`,
/// `"partial_rotary_factor": 0.5`).
const CHATGLM_ROTARY_SHARE: f64 = 0.5;

/// ChatGLM's config, put in Llama's shape: its key-value groups
/// (`multi_query_group_num` under `multi_query_attention`), head width
/// (`kv_channels`), vocabulary (`padded_vocab_size`), the half of each head
/// that rotates, and the base `10000 · rope_ratio`.
fn chatglm_keys(raw: &mut Value) {
    let groups = raw.get("multi_query_group_num").filter(|_| flag(raw, "multi_query_attention")).cloned();
    let heads = raw.get("num_attention_heads").cloned();
    let width = raw.get("kv_channels").cloned();
    let vocabulary = raw.get("padded_vocab_size").cloned();
    let base = DEFAULT_ROPE_THETA * number(raw, "rope_ratio").unwrap_or(1.0);
    let Some(object) = raw.as_object_mut() else {
        return;
    };
    if let Some(key_value_heads) = groups.or(heads) {
        object.entry("num_key_value_heads").or_insert(key_value_heads);
    }
    if let Some(width) = width {
        object.entry("head_dim").or_insert(width);
    }
    if let Some(vocabulary) = vocabulary {
        object.entry("vocab_size").or_insert(vocabulary);
    }
    object.entry("partial_rotary_factor").or_insert(Value::from(CHATGLM_ROTARY_SHARE));
    object.entry("rope_theta").or_insert(Value::from(base));
}

/// Nemotron Puzzle's config, put in Nemotron-H's shape: its
/// `layers_block_type` (`mamba`, `attention`, `mlp`, `moe`) becomes
/// `hybrid_override_pattern` (`M`, `*`, `-`, `E`), and the first `moe`
/// block's `num_experts_per_tok` and `moe_intermediate_size` stand in for
/// the top-level ones the per-layer `block_configs` override. A block type
/// it does not know is left as `?`, which the Nemotron-H reading refuses
/// by layer.
fn nemotron_puzzle_keys(raw: &mut Value) {
    let pattern: Option<String> = raw.get("layers_block_type").and_then(Value::as_array).map(|kinds| {
        kinds
            .iter()
            .map(|kind| match kind.as_str() {
                Some("mamba") => 'M',
                Some("attention") => '*',
                Some("mlp") => '-',
                Some("moe") => 'E',
                _ => '?',
            })
            .collect()
    });
    let first_moe = raw
        .get("block_configs")
        .and_then(Value::as_array)
        .and_then(|blocks| blocks.iter().find(|block| block.get("block_type").and_then(Value::as_str) == Some("moe")))
        .cloned();
    let Some(object) = raw.as_object_mut() else {
        return;
    };
    if let Some(pattern) = pattern {
        object.entry("hybrid_override_pattern").or_insert(Value::from(pattern));
    }
    if let Some(block) = first_moe {
        for key in ["num_experts_per_tok", "moe_intermediate_size"] {
            if let Some(value) = block.get(key).cloned() {
                object.entry(key).or_insert(value);
            }
        }
    }
}

/// Step 3.5's config, put in the shape the rest of the reading expects. Its
/// per-layer lists also cover the multi-token-prediction layers past
/// `num_hidden_layers`, which Ster does not run, so they are cut to the
/// decoder's layers; a list `rope_theta` becomes `layer_rope_theta`, with
/// the first full-attention layer's base as `rope_theta`; and the
/// sliding-window layers' head counts in `attention_other_setting` become
/// the base counts, the full-attention ones moving to
/// `num_global_attention_heads` and `num_global_key_value_heads`.
fn step3p5_keys(raw: &mut Value) {
    let layers = whole(raw, "num_hidden_layers").unwrap_or(0);
    let Some(object) = raw.as_object_mut() else {
        return;
    };
    let per_layer = [
        "layer_types",
        "rope_theta",
        "partial_rotary_factors",
        "swiglu_limits",
        "swiglu_limits_shared",
        "use_rope_layers",
    ];
    for key in per_layer {
        if let Some(Value::Array(list)) = object.get_mut(key) {
            list.truncate(layers);
        }
    }
    if let Some(Value::Array(bases)) = object.get("rope_theta").cloned() {
        let first_full = object
            .get("layer_types")
            .and_then(Value::as_array)
            .and_then(|kinds| kinds.iter().position(|kind| kind.as_str() == Some("full_attention")))
            .unwrap_or(0);
        if let Some(base) = bases.get(first_full).cloned() {
            object.insert("rope_theta".to_owned(), base);
        }
        object.insert("layer_rope_theta".to_owned(), Value::Array(bases));
    }
    let Some(other) = object
        .get("attention_other_setting")
        .filter(|setting| setting.get("attention_type").and_then(Value::as_str) == Some("sliding_attention"))
        .cloned()
    else {
        return;
    };
    let full_heads = object.get("num_attention_heads").cloned();
    let full_groups = object.get("num_attention_groups").or_else(|| object.get("num_key_value_heads")).cloned();
    let windowed_heads = other.get("num_attention_heads").cloned();
    let windowed_groups = other.get("num_attention_groups").cloned();
    if let (Some(heads), Some(groups), Some(own_heads), Some(own_groups)) =
        (full_heads, full_groups, windowed_heads, windowed_groups)
    {
        object.insert("num_global_attention_heads".to_owned(), heads);
        object.insert("num_global_key_value_heads".to_owned(), groups);
        object.insert("num_attention_heads".to_owned(), own_heads);
        object.insert("num_key_value_heads".to_owned(), own_groups);
    }
}

/// Step 3.5 (`step3p5`): RMS norms scaled by `1 + weight` throughout (vLLM's
/// `GemmaRMSNorm`), per-head query and key norms, and a sigmoid head-wise
/// gate (`g_proj`) under `use_head_wise_attn_gate`. Its layers alternate
/// full and `sliding_window` attention as `layer_types` lists; the
/// sliding-window layers' head counts come from `attention_other_setting`,
/// each kind rotates its own `partial_rotary_factors` share by halves at
/// its own base, `rope_scaling` reaches the full-attention layers only
/// (`yarn_only_types`), and `use_rope_layers` can leave layers unrotated.
/// Step's experts route by `moe_router_activation` scores that
/// `moe.router_bias` moves to choose and `moe_router_scaling_factor` scales,
/// each routed and shared expert's SwiGLU clamped on the layers
/// `swiglu_limits` and `swiglu_limits_shared` give a limit.
fn step3p5(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!("{} declares a Step 3.5 model without layer_types", path.display());
    };
    let windowed = listed_layers(types, layers, path)?;
    if let Some(other) = raw.get("attention_other_setting").filter(|setting| !setting.is_null()) {
        let kind = other.get("attention_type").and_then(Value::as_str);
        if kind != Some("sliding_attention") {
            bail!(
                "{} states attention_other_setting for {kind:?} layers; Ster reads it for sliding_attention layers",
                path.display()
            );
        }
        if whole(other, "head_dim").is_some_and(|width| width != architecture.head_dim) {
            bail!(
                "{} gives its sliding-window heads a head_dim other than {}; Ster builds both kinds one width",
                path.display(),
                architecture.head_dim
            );
        }
    }
    let scaled = raw.get("rope_scaling").is_some_and(|scaling| !scaling.is_null());
    let full_only = raw
        .get("yarn_only_types")
        .and_then(Value::as_array)
        .is_some_and(|kinds| kinds.len() == 1 && kinds[0].as_str() == Some("full_attention"));
    if scaled && !full_only {
        bail!(
            "{} scales the rotation of sliding-window layers too (yarn_only_types); Ster scales the full-attention rotation only",
            path.display()
        );
    }
    architecture.sliding_layers = windowed;
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.norm_offset = true;
    if flag(raw, "use_qk_norm") {
        architecture.query_key_norm = QueryKeyNorm::PerHead;
    }
    architecture.head_gate = flag(raw, "use_head_wise_attn_gate").then_some(GateFunction::Sigmoid);
    if let Some(rotated) = raw.get("use_rope_layers").and_then(Value::as_array).filter(|list| !list.is_empty()) {
        if rotated.len() != layers {
            bail!("{} lists {} use_rope_layers entries for {layers} layers", path.display(), rotated.len());
        }
        for (layer, entry) in rotated.iter().enumerate() {
            if entry.as_bool() == Some(false) {
                architecture.unrotated_layers |= 1u128 << layer;
            }
        }
    }
    let head_dim = architecture.head_dim;
    let (full_share, windowed_share) = per_kind(raw, "partial_rotary_factors", windowed, layers, path)?;
    let width = |share: Option<f64>| (head_dim as f64 * share.unwrap_or(1.0)) as usize;
    let (global_rotary, local_rotary) = (width(full_share), width(windowed_share));
    if [global_rotary, local_rotary].iter().any(|rotated| *rotated == 0 || rotated % 2 != 0 || *rotated > head_dim) {
        bail!(
            "{} rotates {global_rotary} and {local_rotary} of {head_dim} components per head (partial_rotary_factors); each must be even, above zero and at most the head",
            path.display()
        );
    }
    architecture.rotary_dim = local_rotary;
    let base = f64::from(llama.rope_theta);
    let (full_base, windowed_base) = per_kind(raw, "layer_rope_theta", windowed, layers, path)?;
    if full_base.is_some_and(|full| full != base) {
        bail!("{} rotates its full-attention layers by more than one base; Ster rotates them all by one", path.display());
    }
    // The sliding-window layers always rotate by their own table: its width
    // can differ from the full-attention layers' and it is never scaled.
    architecture.local_rope_theta = Some(windowed_base.unwrap_or(base) as f32);
    architecture.global_attention = Some(GlobalAttention {
        heads: whole(raw, "num_global_attention_heads"),
        head_dim,
        key_value_heads: whole(raw, "num_global_key_value_heads"),
        rotary_dim: global_rotary,
        key_is_value: false,
    });
    let mut routed = step_experts(raw, layers, path)?;
    routed.scoring = match text(raw, "moe_router_activation") {
        None | Some("sigmoid") => Scoring::Sigmoid,
        Some("softmax") => Scoring::Softmax,
        Some(other) => bail!(
            "{} declares moe_router_activation {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    routed.routed_scale = number(raw, "moe_router_scaling_factor");
    routed.selection_bias = flag(raw, "use_moe_router_bias").then_some("moe.router_bias");
    step_limits(raw, ("swiglu_limits", "swiglu_limits_shared"), layers, &mut routed, path)?;
    architecture.experts = Some(routed);
    Ok(())
}

/// Per-layer SwiGLU clamps on the routed and shared experts (Step 3.5's
/// `swiglu_limits` and `swiglu_limits_shared`, Ling 3.0's
/// `expert_swiglu_limit_list` and `share_expert_swiglu_limit_list`): one
/// entry per layer, zero or null meaning none. A dense layer's shared limit
/// is refused, since Ster clamps expert feed-forwards only.
fn step_limits(
    raw: &Value,
    (routed_key, shared_key): (&str, &str),
    layers: usize,
    routed: &mut MixtureOfExperts,
    path: &Path,
) -> Result<()> {
    let limits = |key: &str| -> Result<Vec<f64>> {
        match raw.get(key).filter(|list| !list.is_null()) {
            None => Ok(Vec::new()),
            Some(list) => list
                .as_array()
                .and_then(|entries| {
                    entries.iter().map(|entry| if entry.is_null() { Some(0.0) } else { entry.as_f64() }).collect()
                })
                .with_context(|| format!("{} declares {key} that is not a list of numbers", path.display())),
        }
    };
    let (routed_limits, shared_limits) = (limits(routed_key)?, limits(shared_key)?);
    if let Some(layer) = (0..layers).find(|layer| {
        routed.dense_layers & (1u128 << layer) != 0 && shared_limits.get(*layer).is_some_and(|limit| *limit != 0.0)
    }) {
        bail!(
            "{} clamps the dense feed-forward of layer {layer} ({shared_key}); Ster clamps expert feed-forwards only",
            path.display()
        );
    }
    if routed_limits.iter().chain(&shared_limits).any(|limit| *limit != 0.0) {
        routed.swiglu_limit = Some(SwigluLimit::Step { routed: routed_limits, shared: shared_limits });
    }
    Ok(())
}

/// A per-layer list of numbers under `key`, one value for the
/// full-attention layers and one for the sliding-window layers in
/// `windowed`, as `(full, sliding)`; refused when either kind's values
/// differ.
fn per_kind(raw: &Value, key: &str, windowed: u128, layers: usize, path: &Path) -> Result<(Option<f64>, Option<f64>)> {
    let Some(list) = raw.get(key).filter(|list| !list.is_null()) else {
        return Ok((None, None));
    };
    let values: Vec<f64> = list
        .as_array()
        .and_then(|entries| entries.iter().map(Value::as_f64).collect())
        .with_context(|| format!("{} declares {key} that is not a list of numbers", path.display()))?;
    if values.len() != layers {
        bail!("{} lists {} {key} values for {layers} layers", path.display(), values.len());
    }
    fits(layers, path)?;
    let mut kinds = [None, None];
    for (layer, value) in values.into_iter().enumerate() {
        let slot = usize::from(windowed & (1u128 << layer) != 0);
        match kinds[slot] {
            Some(seen) if seen != value => bail!(
                "{} lists more than one {key} value for its {} layers; Ster gives each attention kind one",
                path.display(),
                if slot == 1 { "sliding-window" } else { "full-attention" }
            ),
            _ => kinds[slot] = Some(value),
        }
    }
    Ok((kinds[0], kinds[1]))
}

/// K2-Horizon (`k2_horizon`): Llama's block with hidden-width RMS norms
/// over `layernorm_num_groups` equal groups, an elementwise attention gate
/// (`self_attn.gate_proj` through `attention_gate_func`, `silu` or
/// `softplus` with β = ln 2) and biased projections under `attention_bias`.
/// The layers neither in `mlp_only_layers` nor off `decoder_sparse_step`
/// route over `num_experts` experts by `router_score_func` scores that
/// `mlp.gate.bias` moves to choose (`moe_gate_bias`), renormalised under
/// `norm_topk_prob` and scaled by `router_scaling_factor`, beside
/// `num_shared_experts` shared ones; under `mova_num_experts` those layers
/// take each value from `mova_num_experts_per_tok` value experts routed the
/// same way. A rotation narrower than the head (`rope_head_dim`, whose
/// channels the checkpoint interleaves) and per-head query and key norms
/// (`query_key_norm`) are refused.
fn k2_horizon(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let head_dim = architecture.head_dim;
    if let Some(rotated) = whole(raw, "rope_head_dim").filter(|rotated| *rotated != head_dim) {
        bail!(
            "{} rotates {rotated} of {head_dim} channels per head (rope_head_dim), interleaved through the head; Ster rotates whole K2-Horizon heads only",
            path.display()
        );
    }
    if flag(raw, "query_key_norm") {
        bail!(
            "{} declares query_key_norm, one scale per channel of every head; Ster implements K2-Horizon without query and key norms",
            path.display()
        );
    }
    let groups = whole(raw, "layernorm_num_groups").unwrap_or(1).max(1);
    if llama.hidden_size % groups != 0 {
        bail!(
            "{} splits its {} hidden channels into {groups} norm groups unevenly",
            path.display(),
            llama.hidden_size
        );
    }
    architecture.norm_groups = groups;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    architecture.attention_gate = match text(raw, "attention_gate_func") {
        None => None,
        Some("silu") => Some(GateFunction::Silu),
        Some("softplus") => Some(GateFunction::Softplus),
        Some(other) => bail!(
            "{} declares attention_gate_func {other:?}; Ster implements silu and softplus",
            path.display()
        ),
    };
    if whole(raw, "num_experts").unwrap_or(0) == 0 {
        return Ok(());
    }
    let scoring = match text(raw, "router_score_func") {
        None | Some("softmax") => Scoring::Softmax,
        Some("sigmoid") => Scoring::Sigmoid,
        Some(other) => bail!(
            "{} declares router_score_func {other:?}; Ster implements softmax and sigmoid expert scores",
            path.display()
        ),
    };
    let biased = flag(raw, "moe_gate_bias");
    let mut routed = experts(
        raw,
        "num_experts",
        "moe_intermediate_size",
        flag(raw, "norm_topk_prob"),
        ExpertLayout::Qwen,
        qwen_dense_layers(raw, layers, path)?,
        path,
    )?;
    routed.scoring = scoring;
    routed.routed_scale = number(raw, "router_scaling_factor");
    routed.selection_bias = biased.then_some("mlp.gate.bias");
    routed.shared = whole(raw, "num_shared_experts")
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    if let Some(count) = whole(raw, "mova_num_experts").filter(|count| *count > 0) {
        let top_k = whole(raw, "mova_num_experts_per_tok").unwrap_or(0);
        if top_k == 0 || top_k > count {
            bail!(
                "{} routes each value to {top_k} of {count} value experts (mova_num_experts_per_tok); it must be at least one and at most all of them",
                path.display()
            );
        }
        architecture.value_experts = Some(MixtureOfExperts {
            count,
            top_k,
            intermediate: llama.num_key_value_heads * head_dim,
            normalize: top_k > 1,
            shared: None,
            layout: ExpertLayout::Mova,
            dense_layers: routed.dense_layers,
            scoring,
            groups: None,
            selection_bias: biased.then_some("self_attn.v_router.bias"),
            routed_scale: routed.routed_scale,
            swiglu_limit: None,
            identity_experts: 0,
            average_shared: false,
            weight_input: false,
            latent: None,
        });
    }
    architecture.experts = Some(routed);
    Ok(())
}

/// LongCat-Flash-Lite's n-gram embeddings (`longcat_flash_ngram`): tables
/// for orders 2 to `emb_neighbor_num`, `emb_split_num` of each, sized by
/// `ngram_vocab_size_ratio`, their runs restarting after `eos_token_id`
/// (the first, when it is a list; 2, the config class's default, when it
/// is left out).
fn longcat_ngram(raw: &Value, path: &Path) -> Result<NgramSpec> {
    let (Some(ratio), Some(splits), Some(neighbors)) =
        (whole(raw, "ngram_vocab_size_ratio"), whole(raw, "emb_split_num"), whole(raw, "emb_neighbor_num"))
    else {
        bail!(
            "{} declares a LongCat n-gram model without ngram_vocab_size_ratio, emb_split_num or emb_neighbor_num",
            path.display()
        );
    };
    if splits == 0 || neighbors < 2 {
        bail!(
            "{} declares {splits} n-gram splits over neighbourhoods of {neighbors}; Ster needs at least one split and a neighbourhood of two tokens",
            path.display()
        );
    }
    let eos = match raw.get("eos_token_id") {
        None | Some(Value::Null) => LONGCAT_NGRAM_EOS,
        Some(Value::Array(ids)) => ids.first().and_then(Value::as_u64).unwrap_or(LONGCAT_NGRAM_EOS),
        Some(id) => id.as_u64().with_context(|| format!("{} declares an eos_token_id that is not a token", path.display()))?,
    };
    Ok(NgramSpec { ratio, splits, neighbors, eos: eos as u32 })
}

/// The end-of-sequence token LongCat's n-gram config class assumes when a
/// config leaves `eos_token_id` out (`configuration_longcat_ngram.py`,
/// `eos_token_id=2`).
const LONGCAT_NGRAM_EOS: u64 = 2;

/// LongCat-Flash (`longcat_flash`): each of its `num_layers` stored layers
/// is two Ster layers, `input_layernorm.{h}`, DeepSeek's latent attention
/// `self_attn.{h}` (rotating adjacent pairs), `post_attention_layernorm.{h}`
/// and a dense `mlps.{h}` for halves 0 and 1, beside shortcut-connected
/// experts that read the first half's feed-forward input and join the
/// residual at the end of the second. Under `mla_scale_q_lora` and
/// `mla_scale_kv_lora` the latent bottlenecks' normed outputs are
/// multiplied by `sqrt(hidden_size / rank)`. The experts' router
/// (`mlp.router.classifier`) scores `n_routed_experts` experts
/// `expert_ffn_hidden_size` wide and `zero_expert_num` identity experts by
/// softmax, `mlp.router.e_score_correction_bias` moving the choice of
/// `moe_topk`, and scales the chosen weights by `routed_scaling_factor`.
fn longcat_flash(raw: &Value, llama: &LlamaConfig, architecture: &mut Architecture, path: &Path) -> Result<()> {
    let Some(latent) = architecture.latent else {
        bail!("{} declares a LongCat-Flash model without kv_lora_rank", path.display());
    };
    if let Some(method) = text(raw, "attention_method").filter(|method| *method != "MLA") {
        bail!(
            "{} declares attention_method {method:?}; Ster implements LongCat-Flash's MLA",
            path.display()
        );
    }
    let identity = whole(raw, "zero_expert_num").unwrap_or(0);
    if identity > 0 && text(raw, "zero_expert_type").is_some_and(|kind| kind != "identity") {
        bail!(
            "{} declares zero_expert_type {:?}; Ster implements identity zero experts",
            path.display(),
            text(raw, "zero_expert_type")
        );
    }
    architecture.interleaved_rotary = true;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    let hidden = llama.hidden_size as f64;
    let scale = |key: &str, rank: Option<usize>| match rank {
        Some(rank) if flag(raw, key) && rank > 0 => (hidden / rank as f64).sqrt(),
        _ => 1.0,
    };
    architecture.latent_scales = Some((
        scale("mla_scale_q_lora", latent.query_rank),
        scale("mla_scale_kv_lora", Some(latent.key_value_rank)),
    ));
    let mut routed = experts(
        raw,
        "n_routed_experts",
        "expert_ffn_hidden_size",
        flag(raw, "norm_topk_prob"),
        ExpertLayout::LongCat,
        0,
        path,
    )?;
    routed.identity_experts = identity;
    routed.selection_bias = Some("mlp.router.e_score_correction_bias");
    routed.routed_scale = number(raw, "routed_scaling_factor");
    architecture.shortcut_experts = Some(routed);
    Ok(())
}

/// Cohere2-MoE (`cohere2_moe`): Cohere's parallel block (one
/// `input_layernorm` before attention and the feed-forward side by side)
/// with RMS norms, or LayerNorms without a bias when it states no
/// `rms_norm_eps`. Adjacent pairs rotate on the sliding-window layers
/// `layer_types` lists, whose window is `sliding_window + 1` keys as vLLM
/// counts it, and under `prefix_dense_sliding_window_pattern` 1 on the
/// dense prefix too; the other layers apply no rotation. The dense layers
/// (`mlp_layer_types`, else the first `first_k_dense_replace`) are
/// `prefix_dense_intermediate_size` wide; the rest route over
/// `num_experts` experts `intermediate_size` wide by `expert_selection_fn`
/// scores, renormalised under `norm_topk_prob`, beside
/// `num_shared_experts` shared ones whose sum with the routed ones is
/// halved under `shared_expert_combination_strategy` `average`.
fn cohere2_moe(raw: &Value, layers: usize, architecture: &mut Architecture, path: &Path) -> Result<()> {
    fits(layers, path)?;
    if flag(raw, "use_qk_norm") {
        bail!(
            "{} declares use_qk_norm; Ster implements Cohere2-MoE without query and key norms",
            path.display()
        );
    }
    if raw.get("use_gated_activation").and_then(Value::as_bool) == Some(false) {
        bail!(
            "{} declares use_gated_activation false; Ster implements Cohere2-MoE's gated feed-forward",
            path.display()
        );
    }
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!("{} declares a Cohere2-MoE model without layer_types", path.display());
    };
    let windowed = listed_layers(types, layers, path)?;
    architecture.norm = if flag(raw, "layer_norm") { NormKind::Layer { bias: false } } else { NormKind::Rms };
    architecture.parallel = true;
    architecture.interleaved_rotary = true;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    architecture.logits_multiplier = number(raw, "logit_scale");
    architecture.sliding_layers = windowed;
    architecture.sliding_window = whole(raw, "sliding_window").map(|window| window + 1);
    let dense = match raw.get("mlp_layer_types").and_then(Value::as_array) {
        Some(kinds) => {
            if kinds.len() != layers {
                bail!("{} lists {} mlp_layer_types for {layers} layers", path.display(), kinds.len());
            }
            let mut set = 0u128;
            for (layer, kind) in kinds.iter().enumerate() {
                match kind.as_str() {
                    Some("dense") => set |= 1u128 << layer,
                    Some("sparse") => {}
                    other => bail!(
                        "{} declares layer {layer}'s feed-forward as {other:?}; Ster implements dense and sparse",
                        path.display()
                    ),
                }
            }
            set
        }
        None => {
            let first = whole(raw, "first_k_dense_replace").unwrap_or(0).min(layers);
            (0..first).fold(0u128, |set, layer| set | (1u128 << layer))
        }
    };
    let prefix = (0..layers)
        .take_while(|layer| dense & (1u128 << layer) != 0)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let forced = if whole(raw, "prefix_dense_sliding_window_pattern").unwrap_or(1) == 1 { prefix } else { 0 };
    architecture.unrotated_layers = every_layer(layers, path)? & !windowed & !forced;
    let normalize = raw.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true);
    let mut routed = experts(raw, "num_experts", "moe_intermediate_size", normalize, ExpertLayout::Qwen, dense, path)?;
    routed.scoring = match text(raw, "expert_selection_fn") {
        None | Some("softmax") => Scoring::Softmax,
        Some("sigmoid") => Scoring::Sigmoid,
        Some(other) => bail!(
            "{} declares expert_selection_fn {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    routed.shared = whole(raw, "num_shared_experts")
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    routed.average_shared = routed.shared.is_some()
        && match text(raw, "shared_expert_combination_strategy") {
            None | Some("sum") => false,
            Some("average") => true,
            Some(other) => bail!(
                "{} declares shared_expert_combination_strategy {other:?}; Ster implements sum and average",
                path.display()
            ),
        };
    architecture.experts = Some(routed);
    Ok(())
}

/// Laguna (`laguna`): Llama's block with per-head query and key norms and a
/// per-head gate (`g_proj` through softplus under `gating` `per-head`; any
/// other gating is refused). Its layers
/// alternate full and `sliding_window` attention as `layer_types` lists,
/// each kind with its own head count (`num_attention_heads_per_layer`) and
/// rotation (`rope_parameters` per layer type: the full layers' YaRN over
/// their `partial_rotary_factor` share, the sliding ones' plain rotation
/// over theirs), the sliding layers with a learned sink per head
/// (`self_attn.sink`) under `swa_attention_sink_enabled`. The layers
/// `mlp_layer_types` calls `sparse` route over `num_experts` experts by
/// sigmoid scores that `mlp.experts.e_score_correction_bias` moves to
/// choose, renormalised and scaled by `moe_routed_scaling_factor`, the
/// weights applied to the experts' input under
/// `moe_apply_router_weight_on_input`, beside a `shared_expert`.
fn laguna(
    raw: &Value,
    scaling: Option<&Value>,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!("{} declares a Laguna model without layer_types", path.display());
    };
    let windowed = listed_layers(types, layers, path)?;
    if number(raw, "moe_router_logit_softcapping").is_some_and(|cap| cap > 0.0) {
        bail!(
            "{} caps its router logits (moe_router_logit_softcapping); Ster implements Laguna's router without a cap",
            path.display()
        );
    }
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = windowed;
    architecture.query_key_norm = QueryKeyNorm::PerHead;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    match raw.get("gating") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => {}
        Some(Value::Bool(true)) => architecture.head_gate = Some(GateFunction::NaturalSoftplus),
        Some(Value::String(kind)) if kind == "per-head" => architecture.head_gate = Some(GateFunction::NaturalSoftplus),
        Some(other) => bail!(
            "{} declares gating {other}; Ster implements Laguna's per-head gate",
            path.display()
        ),
    }
    if flag(raw, "swa_attention_sink_enabled") {
        architecture.names = Names::LAGUNA;
        architecture.attention_sinks = windowed;
    }
    // Each kind's own head count, as `num_attention_heads_per_layer` lists
    // it; the sliding-window count is the base one.
    let full_heads = match raw.get("num_attention_heads_per_layer").and_then(Value::as_array) {
        Some(counts) => {
            if counts.len() != layers {
                bail!("{} lists {} num_attention_heads_per_layer for {layers} layers", path.display(), counts.len());
            }
            let mut kinds = [None, None];
            for (layer, count) in counts.iter().enumerate() {
                let count = count.as_u64().map(|count| count as usize);
                let slot = usize::from(windowed & (1u128 << layer) != 0);
                match (kinds[slot], count) {
                    (_, None) => bail!("{} lists a head count for layer {layer} that is not a whole number", path.display()),
                    (Some(seen), Some(count)) if seen != count => bail!(
                        "{} gives its {} layers more than one head count; Ster gives each attention kind one",
                        path.display(),
                        if slot == 1 { "sliding-window" } else { "full-attention" }
                    ),
                    (_, count) => kinds[slot] = count,
                }
            }
            if kinds[1].is_some_and(|count| count != llama.num_attention_heads) {
                bail!(
                    "{} states num_attention_heads {} unlike its sliding-window layers' {:?}",
                    path.display(),
                    llama.num_attention_heads,
                    kinds[1]
                );
            }
            kinds[0]
        }
        None => None,
    };
    let head_dim = architecture.head_dim;
    let share = |value: Option<f64>| (head_dim as f64 * value.unwrap_or(1.0)) as usize;
    let global_rotary = share(scaling.and_then(|scaling| scaling.get("partial_rotary_factor")).and_then(Value::as_f64));
    let local_rotary = share(number(raw, "local_partial_rotary_factor"));
    if [global_rotary, local_rotary].iter().any(|rotated| *rotated == 0 || rotated % 2 != 0 || *rotated > head_dim) {
        bail!(
            "{} rotates {global_rotary} and {local_rotary} of {head_dim} components per head; each must be even, above zero and at most the head",
            path.display()
        );
    }
    architecture.rotary_dim = local_rotary;
    architecture.rope_scaling = rope_scaling(scaling, global_rotary, raw, llama, path)?;
    let base = f64::from(llama.rope_theta);
    architecture.local_rope_theta = Some(number(raw, "rope_local_base_freq").unwrap_or(base) as f32);
    architecture.global_attention = Some(GlobalAttention {
        heads: full_heads.filter(|heads| *heads != llama.num_attention_heads),
        head_dim,
        key_value_heads: None,
        rotary_dim: global_rotary,
        key_is_value: false,
    });
    let mut routed = experts(
        raw,
        "num_experts",
        "moe_intermediate_size",
        true,
        ExpertLayout::Qwen,
        qwen_dense_layers(raw, layers, path)?,
        path,
    )?;
    routed.scoring = Scoring::Sigmoid;
    routed.selection_bias = Some("mlp.experts.e_score_correction_bias");
    routed.routed_scale = number(raw, "moe_routed_scaling_factor");
    routed.weight_input = flag(raw, "moe_apply_router_weight_on_input");
    routed.shared = whole(raw, "shared_expert_intermediate_size")
        .filter(|width| *width > 0)
        .map(|intermediate| SharedExpert {
            intermediate,
            module: "mlp.shared_expert",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    architecture.experts = Some(routed);
    Ok(())
}

/// AFMoE (`afmoe`, Arcee Trinity): sandwich norms (`post_attention_layernorm`
/// over attention's output, `pre_mlp_layernorm` before the feed-forward,
/// `post_mlp_layernorm` over it), per-head query and key norms, attention's
/// output multiplied by the sigmoid of `self_attn.gate_proj`, and the
/// rotation on the sliding-window layers `layer_types` lists only. Under
/// `mup_enabled` the embeddings are multiplied by `sqrt(hidden_size)`. The
/// layers from `num_dense_layers` route over `num_experts` experts by
/// `score_func` scores that `mlp.expert_bias` moves to choose (within the
/// best `topk_group` of `n_group` groups), renormalised under `route_norm`
/// when the scores are sigmoids and scaled by `route_scale`, beside
/// `num_shared_experts` shared ones.
fn afmoe(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let windowed = match raw.get("layer_types").and_then(Value::as_array) {
        Some(types) => listed_layers(types, layers, path)?,
        None => {
            let every = whole(raw, "global_attn_every_n_layers").unwrap_or(1).max(1);
            (0..layers)
                .filter(|layer| (layer + 1) % every != 0)
                .fold(0u128, |set, layer| set | (1u128 << layer))
        }
    };
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = windowed;
    architecture.unrotated_layers = every_layer(layers, path)? & !windowed;
    architecture.output_norms = true;
    architecture.names = Names::PANGU_SANDWICH;
    architecture.query_key_norm = QueryKeyNorm::PerHead;
    architecture.attention_gate = Some(GateFunction::Sigmoid);
    if flag(raw, "mup_enabled") {
        architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
    }
    let first = whole(raw, "num_dense_layers").unwrap_or(0).min(layers);
    let dense = (0..first).fold(0u128, |set, layer| set | (1u128 << layer));
    let sigmoid = match text(raw, "score_func") {
        None | Some("sigmoid") => true,
        Some("softmax") => false,
        Some(other) => bail!(
            "{} declares score_func {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    let normalize = sigmoid && raw.get("route_norm").and_then(Value::as_bool).unwrap_or(true);
    let mut routed = experts(raw, "num_experts", "moe_intermediate_size", normalize, ExpertLayout::HyV3, dense, path)?;
    routed.scoring = if sigmoid { Scoring::Sigmoid } else { Scoring::Softmax };
    routed.routed_scale = number(raw, "route_scale");
    routed.selection_bias = Some("mlp.expert_bias");
    routed.groups = match (whole(raw, "n_group"), whole(raw, "topk_group")) {
        (Some(groups), Some(chosen_groups)) if groups > 1 => {
            if routed.count % groups != 0 || chosen_groups > groups {
                bail!(
                    "{} splits {} experts into {groups} groups and keeps {chosen_groups}; the groups must divide the experts evenly and at least as many must exist as are kept",
                    path.display(),
                    routed.count
                );
            }
            Some(ExpertGroups { groups, chosen_groups, rank_by_top_two: true })
        }
        _ => None,
    };
    routed.shared = whole(raw, "num_shared_experts")
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    architecture.experts = Some(routed);
    Ok(())
}

/// How often Llama 4 skips the rotation when its config lists no
/// `no_rope_layers`: every fourth layer, counting from one (Transformers'
/// `Llama4TextConfig`, `no_rope_layer_interval=4`).
const LLAMA4_NOPE_INTERVAL: usize = 4;

/// Llama 4's attention temperature when its config leaves it out:
/// `attn_scale` and `floor_scale` (Transformers' `Llama4TextConfig`
/// defaults, 0.1 and 8192).
const LLAMA4_ATTENTION_SCALE: f64 = 0.1;
const LLAMA4_FLOOR_SCALE: usize = 8192;

/// Llama 4's query temperature as Ministral 3 and Mistral Large 3 state it
/// (vLLM's `llama_4_scaling`): every query past each
/// `original_max_position_embeddings` positions multiplied by
/// `1 + beta · ln(1 + floor(position / original))`, the two read from the
/// `llama_4_scaling` block or, as Transformers writes them, beside the
/// rotation (`llama_4_scaling_beta`). None when no beta is stated.
fn llama4_temperature(raw: &Value, scaling: Option<&Value>, layers: usize, path: &Path) -> Result<Option<QueryTemperature>> {
    let stated = |key: &str| {
        raw.get("llama_4_scaling")
            .and_then(|block| block.get(key))
            .or_else(|| scaling.and_then(|scaling| scaling.get(key)))
            .and_then(Value::as_f64)
    };
    let beta = stated("llama_4_scaling_beta").or_else(|| stated("beta"));
    let original = stated("original_max_position_embeddings").map(|original| original as usize);
    Ok(match (beta, original) {
        (Some(beta), Some(interval)) if interval > 0 => Some(QueryTemperature {
            beta,
            interval,
            shift: 0,
            layers: every_layer(layers, path)?,
        }),
        (Some(_), _) => bail!(
            "{} states llama_4_scaling_beta without original_max_position_embeddings to count positions by",
            path.display()
        ),
        (None, _) => None,
    })
}

/// Llama 4 (`llama4_text`, also inside `llama4`): Llama's attention, the
/// layers `no_rope_layers` marks (every fourth by default) unrotated and
/// attending fully, the rotating ones attending by chunks of
/// `attention_chunk_size` with their queries and keys normed without a
/// weight after the rotation under `use_qk_norm`; under
/// `attn_temperature_tuning` the unrotated layers' queries at position `p`
/// are multiplied by `1 + attn_scale · ln(1 + floor((p + 1) / floor_scale))`.
/// The layers `moe_layers` lists (else every `interleave_moe_layer_step`-th)
/// route each token to `num_experts_per_tok` of `num_local_experts` experts
/// by sigmoid scores that weigh the expert's input, beside a shared expert;
/// the others are dense, `intermediate_size_mlp` wide.
fn llama4(raw: &Value, layers: usize, architecture: &mut Architecture, path: &Path) -> Result<()> {
    let every = every_layer(layers, path)?;
    architecture.names = Names::LLAMA4;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    let rotating = match raw.get("no_rope_layers").and_then(Value::as_array).filter(|flags| !flags.is_empty()) {
        Some(flags) => {
            if flags.len() != layers {
                bail!("{} lists {} no_rope_layers entries for {layers} layers", path.display(), flags.len());
            }
            flags
                .iter()
                .enumerate()
                .filter(|(_, flag)| flag.as_u64() == Some(1) || flag.as_bool() == Some(true))
                .fold(0u128, |set, (layer, _)| set | (1u128 << layer))
        }
        None => (0..layers)
            .filter(|layer| (layer + 1) % LLAMA4_NOPE_INTERVAL != 0)
            .fold(0u128, |set, layer| set | (1u128 << layer)),
    };
    architecture.unrotated_layers = every & !rotating;
    if let Some(chunk) = whole(raw, "attention_chunk_size").filter(|chunk| *chunk > 0) {
        architecture.sliding_window = Some(chunk);
        architecture.sliding_layers = rotating;
        architecture.chunk_lookback = Some(0);
    }
    if flag(raw, "use_qk_norm") {
        architecture.query_key_norm = QueryKeyNorm::Unscaled;
        architecture.norm_after_rotary = true;
    }
    if raw.get("attn_temperature_tuning").and_then(Value::as_bool).unwrap_or(true) {
        architecture.query_temperature = Some(QueryTemperature {
            beta: number(raw, "attn_scale").unwrap_or(LLAMA4_ATTENTION_SCALE),
            interval: whole(raw, "floor_scale").unwrap_or(LLAMA4_FLOOR_SCALE).max(1),
            shift: 1,
            layers: every & !rotating,
        });
    }
    let routed_layers = match raw.get("moe_layers").and_then(Value::as_array) {
        Some(listed) => {
            let mut set = 0u128;
            for entry in listed {
                match entry.as_u64().map(|layer| layer as usize) {
                    Some(layer) if layer < layers => set |= 1u128 << layer,
                    _ => bail!("{} lists moe_layers entry {entry}, which names no layer below {layers}", path.display()),
                }
            }
            set
        }
        None => {
            let step = whole(raw, "interleave_moe_layer_step").unwrap_or(1).max(1);
            (step - 1..layers).step_by(step).fold(0u128, |set, layer| set | (1u128 << layer))
        }
    };
    let mut routed =
        experts(raw, "num_local_experts", "moe_intermediate_size", false, ExpertLayout::Llama4, every & !routed_layers, path)?;
    routed.scoring = Scoring::Sigmoid;
    routed.weight_input = true;
    routed.shared = Some(SharedExpert {
        intermediate: routed.intermediate,
        module: "feed_forward.shared_expert",
        gated: false,
        form: SharedForm::GateUpDown,
    });
    architecture.experts = Some(routed);
    Ok(())
}

/// The routed scale Sarvam's latent model uses when its config leaves
/// `routed_scaling_factor` out (vLLM's `sarvam.py`,
/// `getattr(config, "routed_scaling_factor", 2.5)`).
const SARVAM_ROUTED_SCALE: f64 = 2.5;

/// Sarvam-105B (`sarvam_mla`): DeepSeek's latent attention, rotating
/// adjacent pairs, and a router over `num_experts` experts on the layers
/// from `first_k_dense_replace` (default one) at every `moe_layer_freq`-th:
/// `score_function` scores (sigmoid by default) that
/// `mlp.gate.e_score_correction_bias` moves under
/// `moe_router_enable_expert_bias` (on by default), within the best
/// `topk_group` of `n_group` groups when both are stated, renormalised
/// unless `norm_topk_prob` is false and scaled by `routed_scaling_factor`,
/// beside `num_shared_experts` shared experts
/// (`moe_shared_expert_intermediate_size`, else `moe_intermediate_size`,
/// wide each).
fn sarvam_mla(raw: &Value, layers: usize, architecture: &mut Architecture, path: &Path) -> Result<()> {
    if architecture.latent.is_none() {
        bail!("{} declares a Sarvam MLA model without kv_lora_rank", path.display());
    }
    fits(layers, path)?;
    architecture.interleaved_rotary = true;
    let first = whole(raw, "first_k_dense_replace").unwrap_or(1);
    let frequency = whole(raw, "moe_layer_freq").unwrap_or(1).max(1);
    let dense = (0..layers)
        .filter(|layer| *layer < first || (layer - first) % frequency != 0)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let normalize = raw.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true);
    let mut routed = experts(raw, "num_experts", "moe_intermediate_size", normalize, ExpertLayout::Qwen, dense, path)?;
    routed.scoring = match text(raw, "score_function") {
        None | Some("sigmoid") => Scoring::Sigmoid,
        Some("softmax") => Scoring::Softmax,
        Some(other) => bail!(
            "{} declares score_function {other:?}; Ster implements sigmoid and softmax expert scores",
            path.display()
        ),
    };
    let biased = raw.get("moe_router_enable_expert_bias").and_then(Value::as_bool).unwrap_or(true);
    routed.selection_bias = biased.then_some("mlp.gate.e_score_correction_bias");
    routed.routed_scale = Some(number(raw, "routed_scaling_factor").unwrap_or(SARVAM_ROUTED_SCALE));
    routed.groups = match (whole(raw, "n_group"), whole(raw, "topk_group")) {
        (Some(groups), Some(chosen_groups)) => {
            if groups == 0 || routed.count % groups != 0 || chosen_groups > groups {
                bail!(
                    "{} splits {} experts into {groups} groups and keeps {chosen_groups}; the groups must divide the experts evenly and at least as many must exist as are kept",
                    path.display(),
                    routed.count
                );
            }
            Some(ExpertGroups { groups, chosen_groups, rank_by_top_two: biased })
        }
        _ => None,
    };
    let shared_width = whole(raw, "moe_shared_expert_intermediate_size").unwrap_or(routed.intermediate);
    routed.shared = Some(whole(raw, "num_shared_experts").unwrap_or(1))
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * shared_width,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    architecture.experts = Some(routed);
    Ok(())
}

/// DeciLM rounds every feed-forward width up to a multiple of this
/// (`_find_multiple(intermediate_size, 256)` in vLLM's `nemotron_nas.py`
/// and the checkpoints' `modeling_decilm.py`).
const DECI_WIDTH_MULTIPLE: usize = 256;

/// A DeciLM `ffn` entry's width: its `intermediate_size`, or
/// `2 · ffn_mult · hidden / 3` truncated and rounded up to a multiple of
/// [`DECI_WIDTH_MULTIPLE`].
fn deci_intermediate(ffn: &Value, hidden: usize) -> Option<usize> {
    if let Some(width) = whole(ffn, "intermediate_size") {
        return Some(width);
    }
    let multiplier = number(ffn, "ffn_mult")?;
    let width = (2.0 * multiplier * hidden as f64 / 3.0) as usize;
    Some(width.div_ceil(DECI_WIDTH_MULTIPLE) * DECI_WIDTH_MULTIPLE)
}

/// DeciLM's `block_configs`, one plan per layer.
fn deci_plans(raw: &Value, layers: usize, llama: &LlamaConfig, path: &Path) -> Result<Vec<LayerPlan>> {
    let Some(blocks) = raw.get("block_configs").and_then(Value::as_array) else {
        bail!("{} declares a DeciLM model without block_configs", path.display());
    };
    if blocks.len() != layers {
        bail!("{} lists {} block_configs for {layers} layers", path.display(), blocks.len());
    }
    let stated = |section: &Value, key: &str| section.get(key).is_some_and(|value| !value.is_null() && value != false);
    blocks
        .iter()
        .enumerate()
        .map(|(layer, block)| {
            let (Some(attention), Some(ffn)) = (block.get("attention"), block.get("ffn")) else {
                bail!("{} gives layer {layer} no attention or ffn block config", path.display());
            };
            for (section, name) in [(attention, "attention"), (ffn, "ffn")] {
                for key in ["replace_with_linear", "sparsify"] {
                    if stated(section, key) {
                        bail!("{} sets layer {layer}'s {name} {key}, which Ster does not implement", path.display());
                    }
                }
            }
            for key in ["window_length", "num_sink_tokens"] {
                if stated(attention, key) {
                    bail!("{} sets layer {layer}'s attention {key}, which Ster does not implement", path.display());
                }
            }
            if text(ffn, "hidden_act").is_some_and(|act| Some(act) != text(raw, "hidden_act")) {
                bail!(
                    "{} gives layer {layer} its own feed-forward hidden_act; Ster runs one activation in every layer",
                    path.display()
                );
            }
            let key_value_heads = if flag(attention, "no_op") {
                None
            } else {
                let Some(group) = whole(attention, "n_heads_in_group").filter(|group| *group > 0) else {
                    bail!("{} gives layer {layer}'s attention no n_heads_in_group", path.display());
                };
                if llama.num_attention_heads % group != 0 {
                    bail!(
                        "{} groups layer {layer}'s {} heads by {group}, which does not divide them",
                        path.display(),
                        llama.num_attention_heads
                    );
                }
                Some(llama.num_attention_heads / group)
            };
            let intermediate = if flag(ffn, "no_op") {
                None
            } else {
                let Some(width) = deci_intermediate(ffn, llama.hidden_size) else {
                    bail!("{} gives layer {layer}'s ffn neither ffn_mult nor intermediate_size", path.display());
                };
                Some(width)
            };
            Ok(LayerPlan { key_value_heads, intermediate })
        })
        .collect()
}

/// GraniteSWA's and GraniteMoeSWA's additions to Granite: a learned sink per
/// head (`self_attn.sinks`), sliding-window layers as `layer_types` lists
/// them (every layer but each fourth from the first without it), and
/// `layer_rope_theta`, a base per layer where zero means no rotation. Ster
/// rotates full-attention layers by `rope_theta` and sliding-window layers
/// by one base of their own, so the stated bases must fit that.
fn granite_windows(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    architecture.attention_sinks = every_layer(layers, path)?;
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = match raw.get("layer_types").and_then(Value::as_array) {
        Some(types) => listed_layers(types, layers, path)?,
        None => {
            fits(layers, path)?;
            (0..layers)
                .filter(|layer| layer % GRANITE_SWA_FULL_EVERY != 0)
                .fold(0, |set, layer| set | (1u128 << layer))
        }
    };
    layer_bases(raw, layers, llama, architecture, path)
}

/// `layer_rope_theta`, a base per layer where zero means no rotation
/// (GraniteSWA, MuseGlimmer). Ster rotates full-attention layers by
/// `rope_theta` and sliding-window layers by one base of their own, so the
/// stated bases must fit that.
fn layer_bases(raw: &Value, layers: usize, llama: &LlamaConfig, architecture: &mut Architecture, path: &Path) -> Result<()> {
    let Some(thetas) = raw.get("layer_rope_theta").filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let thetas: Vec<f64> = thetas
        .as_array()
        .and_then(|list| list.iter().map(Value::as_f64).collect())
        .with_context(|| format!("{} declares layer_rope_theta that is not a list of numbers", path.display()))?;
    if thetas.len() != layers {
        bail!("{} lists {} layer_rope_theta values for {layers} layers", path.display(), thetas.len());
    }
    let mut local: Option<f64> = None;
    for (layer, theta) in thetas.into_iter().enumerate() {
        let sliding = architecture.window(layer).is_some();
        if theta == 0.0 {
            architecture.unrotated_layers |= 1u128 << layer;
        } else if !sliding && theta != f64::from(llama.rope_theta) {
            bail!(
                "{} rotates full-attention layer {layer} by base {theta}, not rope_theta {}; Ster rotates every full-attention layer by rope_theta",
                path.display(),
                llama.rope_theta
            );
        } else if sliding && local.is_some_and(|base| base != theta) {
            bail!(
                "{} rotates sliding-window layers by more than one base in layer_rope_theta; Ster rotates them all by one",
                path.display()
            );
        } else if sliding {
            local = Some(theta);
        }
    }
    architecture.local_rope_theta = local.filter(|base| *base != f64::from(llama.rope_theta)).map(|base| base as f32);
    Ok(())
}

/// MuseGlimmer's text decoder (`muse_glimmer_text`, inside `muse_glimmer`,
/// whose weights sit below `model.language_model` and whose `lm_head` sits
/// at the root): Gemma 2's sandwich norms storing their scale as an offset
/// from one, the output norms at `post_norm_eps`, and a final norm storing
/// its scale as is; a weightless RMS norm after the embedding; a weightless
/// per-head norm over every query and key unless `use_qk_norm` is false,
/// the query then multiplied by `scale_query_by` (else `qk_scale_factor`,
/// divided by `sqrt(head_dim)` when it is at least that, as vLLM's
/// `_muse_glimmer_query_prescale` reads both schemas); attention's output
/// multiplied by the sigmoid of `self_attn.gate_proj` unless
/// `use_attn_output_gate` is false; sliding-window layers as `layer_types`
/// lists, the bases of `layer_rope_theta` with zero for the unrotated
/// full-attention layers; and the logits multiplied by `output_multiplier`
/// and capped at `final_logit_softcapping`.
fn muse_glimmer(raw: &Value, layers: usize, llama: &LlamaConfig, architecture: &mut Architecture, path: &Path) -> Result<()> {
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!("{} declares a MuseGlimmer model without layer_types", path.display());
    };
    architecture.names = Names::MUSE_GLIMMER;
    architecture.norm_offset = true;
    architecture.plain_final_norm = true;
    architecture.output_norms = true;
    architecture.output_norm_eps = number(raw, "post_norm_eps");
    architecture.embedding_norm = true;
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = listed_layers(types, layers, path)?;
    layer_bases(raw, layers, llama, architecture, path)?;
    if raw.get("use_qk_norm").and_then(Value::as_bool) != Some(false) {
        architecture.query_key_norm = QueryKeyNorm::Weightless;
        let head_dim = architecture.head_dim as f64;
        let prescale = match (number(raw, "scale_query_by"), number(raw, "qk_scale_factor")) {
            (Some(explicit), _) => explicit,
            (None, Some(factor)) if factor >= head_dim.sqrt() => factor / head_dim.sqrt(),
            (None, Some(factor)) => factor,
            (None, None) => 1.0,
        };
        if prescale <= 0.0 {
            bail!("{} scales its queries by {prescale}; the scale must be above zero", path.display());
        }
        architecture.score_divisor = head_dim.sqrt() / prescale;
    }
    if raw.get("use_attn_output_gate").and_then(Value::as_bool) != Some(false) {
        architecture.attention_gate = Some(GateFunction::Sigmoid);
    }
    architecture.logits_multiplier = number(raw, "output_multiplier");
    architecture.final_softcap = number(raw, "final_logit_softcapping");
    Ok(())
}

/// A list of exactly `N` numbers under `key`.
fn numbers<const N: usize>(raw: &Value, key: &str, path: &Path) -> Result<[f64; N]> {
    raw.get(key)
        .and_then(Value::as_array)
        .and_then(|list| list.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>())
        .and_then(|values| <[f64; N]>::try_from(values).ok())
        .with_context(|| format!("{} declares {key} that is not {N} numbers", path.display()))
}

fn flag(raw: &Value, key: &str) -> bool {
    raw.get(key).and_then(Value::as_bool) == Some(true)
}

fn whole(raw: &Value, key: &str) -> Option<usize> {
    raw.get(key).and_then(Value::as_u64).map(|value| value as usize)
}

fn number(raw: &Value, key: &str) -> Option<f64> {
    raw.get(key).and_then(Value::as_f64)
}

fn text<'a>(raw: &'a Value, key: &str) -> Option<&'a str> {
    raw.get(key).and_then(Value::as_str)
}

/// A layer set needs one bit per layer.
fn fits(layers: usize, path: &Path) -> Result<()> {
    if layers > u128::BITS as usize {
        bail!(
            "{} has {layers} layers with per-layer attention settings; Ster tracks them per layer for at most {} layers",
            path.display(),
            u128::BITS
        );
    }
    Ok(())
}

/// SmolLM3's `no_rope_layers`: one entry per layer, `1` where the layer
/// rotates and `0` where it does not. Returns the layers that do not.
fn unrotated(flags: &[Value], layers: usize, path: &Path) -> Result<u128> {
    if flags.len() != layers {
        bail!(
            "{} lists {} no_rope_layers for {layers} layers",
            path.display(),
            flags.len()
        );
    }
    fits(layers, path)?;
    let mut set = 0u128;
    for (layer, flag) in flags.iter().enumerate() {
        match flag.as_u64() {
            Some(0) => set |= 1u128 << layer,
            Some(1) => {}
            _ => bail!(
                "{} marks layer {layer} in no_rope_layers as {flag}; expected 0 or 1",
                path.display()
            ),
        }
    }
    Ok(set)
}

fn every_layer(layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
    Ok((0..layers).fold(0, |set, layer| set | (1u128 << layer)))
}

fn even_layers(layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
    Ok((0..layers).step_by(2).fold(0, |set, layer| set | (1u128 << layer)))
}

/// `layer_types`, as newer configs list it: one entry per layer, either
/// `sliding_attention` or `full_attention`.
fn listed_layers(types: &[Value], layers: usize, path: &Path) -> Result<u128> {
    windowed_layers(types, layers, false, path)
}

/// [`listed_layers`], with `chunked_attention` entries windowed too when
/// the family attends by chunks (`chunked`).
fn windowed_layers(types: &[Value], layers: usize, chunked: bool, path: &Path) -> Result<u128> {
    if types.len() != layers {
        bail!(
            "{} lists {} layer_types for {layers} layers",
            path.display(),
            types.len()
        );
    }
    fits(layers, path)?;
    let mut set = 0u128;
    for (layer, kind) in types.iter().enumerate() {
        match kind.as_str() {
            Some("sliding_attention") => set |= 1u128 << layer,
            Some("chunked_attention") if chunked => set |= 1u128 << layer,
            Some("full_attention") => {}
            other => bail!(
                "{} declares layer {layer} as {other:?}; Ster implements sliding_attention and full_attention{}",
                path.display(),
                if chunked { " and chunked_attention" } else { "" }
            ),
        }
    }
    Ok(set)
}

/// A per-layer list of ones and zeros under `key` (MiMo-V2's
/// `hybrid_layer_pattern` and `moe_layer_freq`): the layers marked one.
fn flagged_layers(raw: &Value, key: &str, layers: usize, path: &Path) -> Result<u128> {
    let Some(flags) = raw.get(key).and_then(Value::as_array) else {
        bail!("{} lists no {key}", path.display());
    };
    if flags.len() != layers {
        bail!("{} lists {} {key} entries for {layers} layers", path.display(), flags.len());
    }
    fits(layers, path)?;
    let mut set = 0u128;
    for (layer, flag) in flags.iter().enumerate() {
        match flag.as_u64() {
            Some(1) => set |= 1u128 << layer,
            Some(0) => {}
            _ => bail!("{} marks layer {layer} in {key} as {flag}; Ster reads 0 or 1", path.display()),
        }
    }
    Ok(set)
}
