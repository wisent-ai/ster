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
    LatentAttention, MixtureOfExperts, Names, NormKind, ParallelScan, ParameterNorm, Positions, QkvLayout,
    QueryKeyNorm, RopeScaling, Scoring, SharedExpert, SharedForm, ShortConvolution, StateSpaceSpec,
    StructuredSpec,
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
    Gemma,
    Gemma2,
    Gemma3Text,
}

impl Family {
    pub(super) const ALL: [Self; 71] = [
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
        Self::Gemma,
        Self::Gemma2,
        Self::Gemma3Text,
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
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Gemma3Text => "gemma3_text",
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
    let aliases: &[(&str, &[&str])] = &[
        ("rms_norm_eps", &["layer_norm_eps", "norm_epsilon", "norm_eps", "layer_norm_epsilon"]),
        ("hidden_size", &["n_embd", "n_embed", "d_model"]),
        ("num_hidden_layers", &["n_layer", "n_layers", "num_layers"]),
        ("num_attention_heads", &["n_head", "n_heads"]),
        ("num_key_value_heads", &["kv_n_heads", "num_attention_groups"]),
        ("max_position_embeddings", &["n_positions", "max_seq_len", "seq_length", "model_max_length"]),
        ("intermediate_size", &["n_inner", "ffn_dim", "ffn_hidden_size"]),
        ("rope_theta", &["rotary_emb_base"]),
        ("num_experts_per_tok", &["moe_k", "moe_top_k", "moe_topk", "num_experts_per_token"]),
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
        Some(other) => bail!(
            "{} declares hidden_act {other:?} for its {model_type} feed-forward; Ster implements silu, gelu, gelu_pytorch_tanh, gelu_new, gelu_fast, relu and relu2",
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
/// the config states at the top level.
pub(super) fn take_rope_scaling(raw: &mut Value, path: &Path) -> Result<Option<Value>> {
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
        "default" | "linear" | "longrope" | "yarn" => Ok(raw
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
        // once a sequence outgrows `max_position_embeddings`, which Ster's
        // rotary tables never do; within them it is the plain rotation.
        "dynamic" => {
            if let Some(object) = raw.as_object_mut() {
                object.remove("rope_scaling");
            }
            Ok(None)
        }
        other => bail!(
            "{} declares rope_scaling {other:?}; Ster implements llama3, linear, longrope, yarn and dynamic rotary scaling",
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
        "yarn" => {
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
                (Some(mscale), Some(all)) => yarn_mscale(factor, mscale) / yarn_mscale(factor, all),
                _ => stated("attention_factor").unwrap_or_else(|| yarn_mscale(factor, 1.0)),
            };
            Ok(RopeScaling::Yarn {
                factor: factor as f32,
                original,
                beta_fast: stated("beta_fast").unwrap_or(YARN_BETA_FAST) as f32,
                beta_slow: stated("beta_slow").unwrap_or(YARN_BETA_SLOW) as f32,
                attention: attention as f32,
            })
        }
        _ => Ok(RopeScaling::None),
    }
}

/// YaRN's default rotation counts bounding the interpolated band (Peng et
/// al., 2023; Transformers' `beta_fast` and `beta_slow`).
const YARN_BETA_FAST: f64 = 32.0;
const YARN_BETA_SLOW: f64 = 1.0;

/// YaRN's attention temperature for a context stretched by `factor`:
/// `0.1 * mscale * ln(factor) + 1`, or one when nothing is stretched.
fn yarn_mscale(factor: f64, mscale: f64) -> f64 {
    const SLOPE: f64 = 0.1;
    if factor <= 1.0 {
        1.0
    } else {
        SLOPE * mscale * factor.ln() + 1.0
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
            let temperature = yarn_mscale(f64::from(*factor), all);
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
        "qwen2" | "qwen3" | "qwen2_moe" | "qwen3_moe" | "qwen3_next" => {
            if model_type.starts_with("qwen2") {
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
                });
            }
            if model_type.ends_with("_moe") || model_type == "qwen3_next" {
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
        "granite" | "granitemoe" | "granitemoehybrid" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.embedding_multiplier = number(raw, "embedding_multiplier");
            architecture.residual_multiplier = number(raw, "residual_multiplier");
            if let Some(multiplier) = number(raw, "attention_multiplier") {
                architecture.score_divisor = 1.0 / multiplier;
            }
            architecture.logits_multiplier = number(raw, "logits_scaling").map(|scale| 1.0 / scale);
            if model_type == "granitemoe" {
                // GraniteMoE takes the softmax over the top-k logits, which is
                // the full softmax renormalised over the chosen experts.
                architecture.experts = Some(experts(
                    raw,
                    "num_local_experts",
                    "intermediate_size",
                    true,
                    ExpertLayout::Granite,
                    0,
                    path,
                )?);
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
        "olmo2" | "olmo3" => {
            architecture.query_key_norm = QueryKeyNorm::Full;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.pre_norms = false;
            architecture.output_norms = true;
            if model_type == "olmo3" {
                architecture.sliding_window = whole(raw, "sliding_window");
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
        "bailing_moe" => {
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
            if flag(raw, "use_qk_norm") {
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
        "deepseek_v2" | "deepseek_v3" | "minicpm3" => {
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
        "gemma" | "gemma2" | "gemma3_text" => {
            architecture.norm_offset = true;
            architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
            architecture.activation = Activation::GeluTanh;
            if model_type != "gemma" {
                architecture.output_norms = true;
                architecture.names = Names::GEMMA2;
                architecture.sliding_window = whole(raw, "sliding_window");
                if let Some(scalar) = number(raw, "query_pre_attn_scalar") {
                    architecture.score_divisor = scalar.sqrt();
                }
                architecture.attention_softcap = number(raw, "attn_logit_softcapping");
                architecture.final_softcap = number(raw, "final_logit_softcapping");
            }
            if model_type == "gemma2" {
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
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    // LFM2's `layer_types` say which layers convolve, read above; every other
    // family's say which layers attend through the window.
    let windows = raw.get("layer_types").and_then(Value::as_array).filter(|_| model_type != "lfm2");
    if let Some(types) = windows {
        architecture.sliding_layers = listed_layers(types, layers, path)?;
    }
    // Cohere 2's global layers apply no rotary embedding; so do EXAONE 4's
    // when the model mixes local and global layers at all.
    let hybrid_exaone = model_type == "exaone4" && architecture.sliding_window.is_some();
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
    let dense_layers = (0..layers)
        .filter(|layer| *layer < first_dense || layer % frequency != 0)
        .fold(0, |set, layer| set | (1u128 << layer));
    let normalize = flag(raw, "norm_topk_prob");
    let mut routed = experts(
        raw,
        "n_routed_experts",
        "moe_intermediate_size",
        normalize,
        ExpertLayout::Qwen,
        dense_layers,
        path,
    )?;
    routed.shared = whole(raw, "n_shared_experts")
        .filter(|shared| *shared > 0)
        .map(|shared| SharedExpert {
            intermediate: shared * routed.intermediate,
            module: "mlp.shared_experts",
            gated: false,
            form: SharedForm::GateUpDown,
        });
    // GLM-4-MoE's and Nemotron-H's routers are DeepSeek-V3's and their
    // configs leave the method out: sigmoid scores, `noaux_tc` selection.
    let v3_default = matches!(model_type, "glm4_moe" | "nemotron_h");
    let v3_router = v3_default || model_type == "deepseek_v3";
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
    routed.selection_bias = (method == "noaux_tc").then_some("mlp.gate.e_score_correction_bias");
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

/// Qwen MoE's dense layers: those in `mlp_only_layers`, and those whose
/// position is not a multiple of `decoder_sparse_step`.
fn qwen_dense_layers(raw: &Value, layers: usize, path: &Path) -> Result<u128> {
    fits(layers, path)?;
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
