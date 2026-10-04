//! Which decoder family a checkpoint is, read from its `config.json`.
//!
//! Every switch below is a key the checkpoint publishes; none is a value Ster
//! picks. A config that asks for something the decoder does not implement is
//! refused with the key that asked for it, instead of being loaded wrong.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{
    Activation, Architecture, ExpertLayout, FeedForwardKind, MixtureOfExperts, Names, NormKind,
    Positions, QkvLayout, QueryKeyNorm, RopeScaling,
};

/// The `model_type` values the decoder implements.
pub(super) const FAMILIES: &[&str] = &[
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "phi",
    "phi3",
    "granite",
    "granitemoe",
    "stablelm",
    "starcoder2",
    "cohere",
    "cohere2",
    "nemotron",
    "olmo",
    "olmo2",
    "olmo3",
    "olmoe",
    "exaone4",
    "internlm3",
    "seed_oss",
    "arcee",
    "ernie4_5",
    "minicpm",
    "orion",
    "glm",
    "glm4",
    "gpt_neox",
    "gptj",
    "smollm3",
    "gpt2",
    "gpt_bigcode",
    "opt",
    "bloom",
    "gemma",
    "gemma2",
    "gemma3_text",
];

/// Families whose Transformers config class leaves `tie_word_embeddings` at
/// the library default, true, so their configs often omit it.
pub(super) const TIED_BY_DEFAULT: &[&str] = &[
    "gemma",
    "gemma2",
    "gemma3_text",
    "cohere",
    "cohere2",
    "starcoder2",
    "ernie4_5",
    "gpt2",
    "gpt_bigcode",
    "opt",
    "bloom",
];

/// OPT's learned position table keeps this many rows before position zero;
/// Transformers' `OPTLearnedPositionalEmbedding` reads `position + offset`.
const OPT_POSITION_OFFSET: usize = 2;

/// OLMo 1's configs state no norm epsilon; Transformers' `OlmoLayerNorm`
/// passes this one to `F.layer_norm` itself.
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
    let aliases: &[(&str, &[&str])] = &[
        ("rms_norm_eps", &["layer_norm_eps", "norm_epsilon", "norm_eps", "layer_norm_epsilon"]),
        ("hidden_size", &["n_embd", "n_embed"]),
        ("num_hidden_layers", &["n_layer"]),
        ("num_attention_heads", &["n_head"]),
        ("max_position_embeddings", &["n_positions"]),
        ("intermediate_size", &["n_inner", "ffn_dim"]),
        ("rope_theta", &["rotary_emb_base"]),
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
    if model_type == "olmo" && missing(raw, "rms_norm_eps") {
        defaults.push(("rms_norm_eps", Value::from(OLMO_NORM_EPS)));
    }
    let four_times = matches!(model_type, "gptj" | "gpt2" | "gpt_bigcode" | "bloom");
    if four_times && missing(raw, "intermediate_size") {
        if let Some(hidden) = raw.get("hidden_size").and_then(Value::as_u64) {
            defaults.push(("intermediate_size", Value::from(GPT_INNER_PER_HIDDEN * hidden)));
        }
    }
    if model_type == "gpt_bigcode" && flag(raw, "multi_query") {
        defaults.push(("num_key_value_heads", Value::from(1u64)));
    }
    if let Some(object) = raw.as_object_mut() {
        for (key, value) in defaults {
            object.insert(key.to_owned(), value);
        }
    }
}

/// The feed-forward non-linearity the config names, as `hidden_act`,
/// `hidden_activation` or `activation_function`; SiLU when it names none.
fn activation(raw: &Value, model_type: &str, path: &Path) -> Result<Activation> {
    let name = text(raw, "hidden_act")
        .or_else(|| text(raw, "hidden_activation"))
        .or_else(|| text(raw, "activation_function"));
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
pub(super) fn take_rope_scaling(raw: &mut Value, path: &Path) -> Result<Option<Value>> {
    let Some(scaling) = raw.get("rope_scaling").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    match scaling_kind(scaling) {
        "llama3" => Ok(None),
        "default" | "linear" | "longrope" => Ok(raw
            .as_object_mut()
            .and_then(|object| object.remove("rope_scaling"))),
        other => bail!(
            "{} declares rope_scaling {other:?}; Ster implements llama3, linear and longrope rotary scaling",
            path.display()
        ),
    }
}

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
            Ok(RopeScaling::LongRope {
                short,
                long,
                original,
                attention: attention as f32,
            })
        }
        _ => Ok(RopeScaling::None),
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
    // The rotated share of each head: `partial_rotary_factor` (or GPT-NeoX's
    // `rotary_pct`) as a fraction, which Transformers truncates, or GPT-J's
    // `rotary_dim` as a width.
    let head_dim = architecture.head_dim;
    let fraction = number(raw, "partial_rotary_factor").or_else(|| number(raw, "rotary_pct"));
    architecture.rotary_dim = match (fraction, whole(raw, "rotary_dim")) {
        (Some(factor), _) => (head_dim as f64 * factor) as usize,
        (None, Some(width)) => width,
        (None, None) => head_dim,
    };
    let rotary_dim = architecture.rotary_dim;
    if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        bail!(
            "{} rotates {rotary_dim} of {head_dim} components per head (partial_rotary_factor, rotary_pct or rotary_dim); the rotated width must be even, above zero and at most the head",
            path.display()
        );
    }
    architecture.rope_scaling = rope_scaling(scaling, architecture.rotary_dim, raw, llama, path)?;
    // Gemma's configs name `gelu` but Transformers runs the tanh
    // approximation for every Gemma, so the family decides, not the key.
    if !model_type.starts_with("gemma") && model_type != "bloom" {
        architecture.activation = activation(raw, model_type, path)?;
    }
    match model_type {
        "llama" | "mistral" | "mixtral" | "phi3" => {
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
        }
        "qwen2" | "qwen3" | "qwen2_moe" | "qwen3_moe" => {
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
            if model_type.ends_with("_moe") {
                let mut routed = experts(
                    raw,
                    "num_experts",
                    "moe_intermediate_size",
                    flag(raw, "norm_topk_prob"),
                    ExpertLayout::Qwen,
                    qwen_dense_layers(raw, layers, path)?,
                    path,
                )?;
                if model_type == "qwen2_moe" {
                    routed.shared_intermediate = whole(raw, "shared_expert_intermediate_size");
                }
                architecture.experts = Some(routed);
            }
        }
        "granite" | "granitemoe" => {
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
        }
        "olmoe" => {
            if raw.get("clip_qkv").is_some_and(|clip| !clip.is_null()) {
                bail!(
                    "{} declares clip_qkv; Ster implements OLMoE without clipping query, key and value",
                    path.display()
                );
            }
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
            if raw.get("clip_qkv").is_some_and(|clip| !clip.is_null()) {
                bail!(
                    "{} declares clip_qkv; Ster implements OLMo without clipping query, key and value",
                    path.display()
                );
            }
            // OLMo 1's LayerNorm stores neither a scale nor a bias.
            architecture.norm = NormKind::Bare;
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
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
        "ernie4_5" => {
            let bias = flag(raw, "use_bias");
            architecture.query_key_value_bias = bias;
            architecture.output_bias = bias;
            architecture.feed_forward_bias = bias;
            architecture.interleaved_rotary = true;
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
        "gpt_neox" => {
            // Pythia and GPT-NeoX: LayerNorm with bias, a head-interleaved
            // `query_key_value`, biased projections and a plain GELU
            // feed-forward; parallel blocks by default, each half with its own
            // norm.
            architecture.names = Names::GPT_NEOX;
            architecture.norm = NormKind::Layer { bias: true };
            architecture.qkv_layout = QkvLayout::HeadInterleaved;
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
            architecture.positions = Positions::Alibi;
            architecture.qkv_layout = QkvLayout::HeadInterleaved;
            architecture.query_key_value_bias = true;
            architecture.output_bias = true;
            architecture.feed_forward_bias = true;
            architecture.feed_forward = FeedForwardKind::Plain;
            architecture.activation = Activation::GeluTanh;
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
    if let Some(types) = raw.get("layer_types").and_then(Value::as_array) {
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
/// `width_key`.
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
        whole(raw, count_key),
        whole(raw, "num_experts_per_tok"),
        whole(raw, width_key),
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
        shared_intermediate: None,
        layout,
        dense_layers,
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
