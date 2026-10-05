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
    GlobalAttention, IndexerSpec, LatentAttention, LayerPlan, LightningForm, LightningSpec, Loops, MixtureOfExperts,
    Names, NormKind, ParallelScan, ParameterNorm,
    PerLayerInputSpec, Positions, QkvLayout, QueryKeyNorm, RopeScaling, ScaledResiduals, Scoring, SharedBlocksSpec,
    SharedExpert, SharedForm, ShortConvolution, SkipConnections, StateSpaceSpec, StructuredSpec,
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
}

impl Family {
    pub(super) const ALL: [Self; 102] = [
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
    ];

    /// The family a config's `model_type` names, if Ster implements it.
    pub(super) fn of(model_type: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|family| family.model_type() == model_type)
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
    let aliases: &[(&str, &[&str])] = &[
        ("rms_norm_eps", &["layer_norm_eps", "norm_epsilon", "norm_eps", "layer_norm_epsilon"]),
        ("hidden_size", &["n_embd", "n_embed", "d_model"]),
        ("num_hidden_layers", &["n_layer", "n_layers", "num_layers"]),
        ("num_attention_heads", &["n_head", "n_heads"]),
        ("num_key_value_heads", &["kv_n_heads", "num_attention_groups"]),
        ("head_dim", &["attention_head_dim"]),
        ("max_position_embeddings", &["n_positions", "max_seq_len", "seq_length", "model_max_length"]),
        ("intermediate_size", &["n_inner", "ffn_dim", "ffn_hidden_size"]),
        ("rope_theta", &["rotary_emb_base"]),
        ("num_experts_per_tok", &["moe_k", "moe_top_k", "moe_topk", "num_experts_per_token", "top_k_experts"]),
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
        "default" | "linear" | "longrope" | "yarn" | "telechat3-yarn" | "proportional" => Ok(raw
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
            "{} declares rope_scaling {other:?}; Ster implements llama3, linear, longrope, yarn, telechat3-yarn, proportional and dynamic rotary scaling",
            path.display()
        ),
    }
}

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
        // slope (its `_compute_telechat_yarn_parameters`).
        kind @ ("yarn" | "telechat3-yarn") => {
            let slope = if kind == "yarn" { YARN_SLOPE } else { TELECHAT3_YARN_SLOPE };
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
        "llama" | "mistral" | "mixtral" | "phi3" | "phimoe" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
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
        "qwen2" | "qwen3" | "qwen2_moe" | "qwen3_moe" | "qwen3_next" | "mimo" | "mellum" => {
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
            if model_type == "qwen3_next" {
                // Qwen3-Next: norms that store their scale as an offset from
                // one, a sigmoid gate on attention's output from `q_proj`,
                // and gated delta-rule linear attention on every layer
                // `layer_types` calls `linear_attention` (without it, on
                // every layer but each `full_attention_interval`-th).
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
                architecture.output_gate = true;
                architecture.delta_rule = Some(DeltaRuleSpec {
                    key_heads: size("linear_num_key_heads")?,
                    value_heads: size("linear_num_value_heads")?,
                    key_dim: size("linear_key_head_dim")?,
                    value_dim: size("linear_value_head_dim")?,
                    kernel: size("linear_conv_kernel_dim")?,
                    layers: linear,
                    form: DeltaRuleForm::Qwen3Next,
                    negative_eigenvalues: false,
                });
            }
            // Mellum is Qwen3-MoE with sliding-window layers, a rotation per
            // layer kind, and dense layers where `mlp_layer_types` says so.
            if model_type.ends_with("_moe") || model_type == "qwen3_next" || model_type == "mellum" {
                let mut routed = experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    flag(raw, "norm_topk_prob"),
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
        "nemotron_h" => {
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
                // `mixer.shared_experts`, `moe_shared_expert_intermediate_size`
                // wide; every projection is `up_proj`/`down_proj`.
                if raw.get("moe_latent_size").is_some_and(|size| !size.is_null()) {
                    bail!(
                        "{} projects its experts' input into moe_latent_size; Ster implements Nemotron-H experts on the full hidden width",
                        path.display()
                    );
                }
                let mut routed = deepseek_experts(raw, model_type, layers, path)?;
                routed.layout = ExpertLayout::NemotronH;
                routed.dense_layers = every_layer(layers, path)? & !routed_layers;
                routed.selection_bias = Some("mixer.gate.e_score_correction_bias");
                routed.shared = whole(raw, "moe_shared_expert_intermediate_size")
                    .filter(|width| *width > 0)
                    .map(|intermediate| SharedExpert {
                        intermediate,
                        module: "mixer.shared_experts",
                        gated: false,
                        form: SharedForm::UpDown,
                    });
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
            architecture.names = Names::NEMOTRON_H;
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
        // Param2-MoE is Ling 2.0's block and router under its own name;
        // Ling 2.5 and 3.0 (`bailing_hybrid`) add lightning attention.
        "bailing_moe" | "param2moe" | "bailing_hybrid" => {
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
            architecture.attention_sinks = true;
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
            routed.swiglu_limit = number(raw, "swiglu_limit");
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
        "deepseek_v2" | "deepseek_v3" | "minicpm3" | "glm4_moe_lite" | "deepseek_v32" | "glm_moe_dsa" | "axk1" => {
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
            if matches!(model_type, "deepseek_v32" | "glm_moe_dsa") {
                architecture.sparse_index = Some(sparse_index(raw, model_type, layers, path)?);
            }
            // A.X-K1 normalises each routed layer's feed-forward output by
            // `post_mlp_layernorm` before the residual add.
            if model_type == "axk1" {
                architecture.names = Names::AXK1;
                architecture.routed_output_norm = true;
            }
        }
        "step3_text" => {
            // Step3 (its text decoder, `text_config` of `step3_vl`): the
            // query narrows through `q_proj` to `share_q_dim`, passes the RMS
            // norm `inter_norm` and widens through `wq`, beside
            // `num_attention_groups` key-value heads; the layers
            // `moe_layers_enum` lists (from zero; all but the first when
            // absent) route over `moe_num_experts` experts stacked under
            // `moe`, renormalised under `norm_expert_weight`, beside a shared
            // `share_expert` `share_expert_dim` wide.
            let Some(width) = whole(raw, "share_q_dim").filter(|width| *width > 0) else {
                bail!("{} declares a Step3 model without share_q_dim", path.display());
            };
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
            architecture.names = Names::STEP3;
            architecture.query_bottleneck = Some(width);
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
            architecture.experts = Some(routed);
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
                head_dim: global_head_dim,
                // Transformers gives the full-attention layers their own
                // key-value head count only under `attention_k_eq_v`.
                key_value_heads: whole(raw, "num_global_key_value_heads").filter(|_| key_is_value),
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
        architecture.sliding_layers = listed_layers(types, layers, path)?;
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
    let v3_default = matches!(model_type, "glm4_moe" | "glm4_moe_lite" | "glm_moe_dsa" | "nemotron_h" | "exaone_moe");
    let v3_router = v3_default || matches!(model_type, "deepseek_v3" | "deepseek_v32" | "axk1");
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
/// `layer_group_size`-th layer, and lightning attention on the others, its
/// heads `head_dim` wide with their first `partial_rotary_factor` share
/// rotated by halves at the latent attention's base.
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
    architecture.attention_sinks = true;
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
            Some("full_attention") => {}
            other => bail!(
                "{} declares layer {layer} as {other:?}; Ster implements sliding_attention and full_attention",
                path.display()
            ),
        }
    }
    Ok(set)
}
