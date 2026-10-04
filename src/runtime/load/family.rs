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
    QueryKeyNorm, RopeScaling,
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
    "olmo2",
    "olmo3",
    "olmoe",
    "smollm3",
    "gemma",
    "gemma2",
    "gemma3_text",
];

/// Families whose Transformers config class leaves `tie_word_embeddings` at
/// the library default, true, so their configs often omit it.
pub(super) const TIED_BY_DEFAULT: &[&str] =
    &["gemma", "gemma2", "gemma3_text", "cohere", "cohere2", "starcoder2"];

/// Candle's Llama config reads the norm epsilon as `rms_norm_eps`; the
/// LayerNorm families spell it otherwise. The first spelling present is
/// copied under Candle's name before the config is parsed.
pub(super) fn fill_norm_eps(raw: &mut Value) {
    if raw.get("rms_norm_eps").is_some() {
        return;
    }
    let spelled = ["layer_norm_eps", "norm_epsilon", "norm_eps", "layer_norm_epsilon"]
        .iter()
        .find_map(|key| raw.get(*key).cloned());
    if let (Some(eps), Some(object)) = (spelled, raw.as_object_mut()) {
        object.insert("rms_norm_eps".to_owned(), eps);
    }
}

/// The feed-forward non-linearity the config names, as `hidden_act` or
/// `hidden_activation`; SiLU when it names none.
fn activation(raw: &Value, model_type: &str, path: &Path) -> Result<Activation> {
    let name = text(raw, "hidden_act").or_else(|| text(raw, "hidden_activation"));
    match name {
        None | Some("silu") | Some("swish") => Ok(Activation::Silu),
        Some("gelu_pytorch_tanh") | Some("gelu_new") | Some("gelu_fast") => Ok(Activation::GeluTanh),
        Some("gelu") => Ok(Activation::Gelu),
        Some("relu2") => Ok(Activation::Relu2),
        Some(other) => bail!(
            "{} declares hidden_act {other:?} for its {model_type} feed-forward; Ster implements silu, gelu, gelu_pytorch_tanh, gelu_new, gelu_fast and relu2",
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
    architecture.rotary_dim = architecture.head_dim;
    if let Some(factor) = number(raw, "partial_rotary_factor").filter(|factor| *factor != 1.0) {
        // Transformers truncates the product, then rotates halves of it.
        let rotary_dim = (architecture.head_dim as f64 * factor) as usize;
        if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > architecture.head_dim {
            bail!(
                "{} rotates {rotary_dim} of {} components per head (partial_rotary_factor {factor}); the rotated width must be even, above zero and at most the head",
                path.display(),
                architecture.head_dim
            );
        }
        architecture.rotary_dim = rotary_dim;
    }
    architecture.rope_scaling = rope_scaling(scaling, architecture.rotary_dim, raw, llama, path)?;
    // Gemma's configs name `gelu` but Transformers runs the tanh
    // approximation for every Gemma, so the family decides, not the key.
    if !model_type.starts_with("gemma") {
        architecture.activation = activation(raw, model_type, path)?;
    }
    match model_type {
        "llama" | "mistral" | "mixtral" | "phi3" => {
            architecture.query_key_value_bias = flag(raw, "attention_bias");
            architecture.output_bias = architecture.query_key_value_bias;
            architecture.fused_projections = model_type == "phi3";
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
            architecture.names = Names::NEMOTRON;
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
        "gemma" | "gemma2" | "gemma3_text" => {
            architecture.norm_offset = true;
            architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
            architecture.activation = Activation::GeluTanh;
            if model_type != "gemma" {
                architecture.output_norms = true;
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
    if model_type == "cohere2" {
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
