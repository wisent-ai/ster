//! Mistral's own checkpoint format: `params.json` beside
//! `consolidated*.safetensors`, as Mistral Large 3 alone is published. The
//! config is put in the Transformers shape the rest of loading reads, and
//! the decoder's tensor names are answered from Mistral's, following vLLM's
//! `transformers_utils/configs/mistral.py` and the weight mappers of its
//! `llama.py` and `mistral_large_3.py`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

/// How a checkpoint names its config and weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `config.json` and the Transformers tensor names.
    Transformers,
    /// `params.json`, `consolidated*.safetensors` and Mistral's tensor names.
    Mistral,
}

/// The context a Mistral config without `max_position_embeddings` gets
/// (vLLM's `_remap_general_mistral_args`, `128_000`).
const MISTRAL_DEFAULT_CONTEXT: u64 = 128_000;

/// What `apply_scale: false` makes YaRN's magnitude correction: none, which
/// Transformers spells as an attention factor of one (vLLM's
/// `_remap_mistral_yarn_args`).
const UNSCALED_ATTENTION: f64 = 1.0;

/// The `mscale_all_dim` vLLM gives every Mistral YaRN rotation.
const MISTRAL_YARN_MSCALE_ALL_DIM: f64 = 1.0;

impl Layout {
    /// The config file this layout reads.
    pub fn config_file(self) -> &'static str {
        match self {
            Self::Transformers => "config.json",
            Self::Mistral => "params.json",
        }
    }

    /// The weight file a merge writes, named so the merged directory
    /// resolves under the same layout.
    pub fn merged_weights(self) -> &'static str {
        match self {
            Self::Transformers => "model.safetensors",
            Self::Mistral => "consolidated.safetensors",
        }
    }

    /// Whether a published safetensors file holds this layout's weights. A
    /// repository may publish both layouts (Mistral 7B v0.3); each reads its
    /// own files only, so the weights are never mapped twice.
    pub fn owns(self, file_name: &str) -> bool {
        let consolidated = file_name.starts_with("consolidated");
        file_name.ends_with(".safetensors")
            && !file_name.contains("optimizer")
            && !file_name.contains("training_args")
            && consolidated == (self == Self::Mistral)
    }

    /// The name a tensor the decoder asks for by its Transformers name is
    /// stored under.
    pub fn stored_name(self, name: &str) -> String {
        match self {
            Self::Transformers => name.to_owned(),
            Self::Mistral => stored_name(name),
        }
    }
}

/// `params` put in the Transformers shape. Mistral Large 3 (latent
/// attention and experts beside shared ones) becomes `deepseek_v3` with
/// softmax scores, renormalised; latent attention without experts becomes a
/// dense `deepseek_v3`; a dense model becomes `mistral`, rotating adjacent
/// pairs as Mistral's own weights are laid out. Experts without shared ones
/// (Mixtral) are refused: their Mistral-format names are not read.
pub fn transformers_config(params: Value, path: &Path) -> Result<Value> {
    let Value::Object(mut config) = params else {
        bail!("{} is not a JSON object", path.display());
    };
    rename(&mut config, "dim", "hidden_size");
    rename(&mut config, "norm_eps", "rms_norm_eps");
    rename(&mut config, "n_kv_heads", "num_key_value_heads");
    rename(&mut config, "n_layers", "num_hidden_layers");
    rename(&mut config, "n_heads", "num_attention_heads");
    rename(&mut config, "hidden_dim", "intermediate_size");
    let activation = config.remove("activation").unwrap_or(Value::from("silu"));
    config.insert("hidden_act".to_owned(), activation);
    let tied = config.remove("tied_embeddings").unwrap_or(Value::Bool(false));
    config.insert("tie_word_embeddings".to_owned(), tied);
    config.entry("max_position_embeddings").or_insert(Value::from(MISTRAL_DEFAULT_CONTEXT));
    if let Some(quantization) = config.remove("quantization") {
        config.insert("quantization_config".to_owned(), quantization);
    }
    sliding_window(&mut config, path)?;
    let latent = config.get("qk_nope_head_dim").is_some_and(|width| !width.is_null());
    let model_type = match config.remove("moe") {
        Some(Value::Object(moe)) => {
            if moe.get("num_shared_experts").and_then(Value::as_u64).unwrap_or(0) == 0 {
                bail!(
                    "{} declares experts without shared ones (Mixtral's layout); Ster reads Mistral's own format for dense models and Mistral Large 3 only, so use the Transformers conversion",
                    path.display()
                );
            }
            experts(&mut config, moe);
            "deepseek_v3"
        }
        Some(Value::Null) | None if latent => "deepseek_v3",
        Some(Value::Null) | None => {
            config.insert("rope_interleave".to_owned(), Value::Bool(true));
            "mistral"
        }
        Some(other) => bail!("{} declares moe {other}, which is not an object", path.display()),
    };
    config.insert("model_type".to_owned(), Value::from(model_type));
    if let Some(yarn) = config.remove("yarn").filter(|yarn| !yarn.is_null()) {
        rope_yarn(&mut config, &yarn, path)?;
    }
    Ok(Value::Object(config))
}

fn rename(config: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(value) = config.remove(from) {
        config.insert(to.to_owned(), value);
    }
}

/// A `sliding_window` list (a window or null per layer, repeated over the
/// layers) becomes `layer_types` and one window; a single window covers
/// every layer.
fn sliding_window(config: &mut Map<String, Value>, path: &Path) -> Result<()> {
    let layers = config.get("num_hidden_layers").and_then(Value::as_u64).unwrap_or(0) as usize;
    match config.get("sliding_window").cloned() {
        None | Some(Value::Null) | Some(Value::Number(_)) => {}
        Some(Value::Array(pattern)) => {
            if pattern.is_empty() || layers % pattern.len() != 0 {
                bail!("{} repeats a sliding_window pattern of {} over {layers} layers unevenly", path.display(), pattern.len());
            }
            let windows: Vec<u64> = pattern.iter().filter_map(Value::as_u64).collect();
            if windows.windows(2).any(|pair| pair[0] != pair[1]) {
                bail!("{} lists more than one sliding window; Ster gives every windowed layer one", path.display());
            }
            let kinds: Vec<Value> = (0..layers)
                .map(|layer| {
                    let kind = if pattern[layer % pattern.len()].is_null() { "full_attention" } else { "sliding_attention" };
                    Value::from(kind)
                })
                .collect();
            config.insert("layer_types".to_owned(), Value::Array(kinds));
            config.insert("sliding_window".to_owned(), windows.first().map_or(Value::Null, |window| Value::from(*window)));
        }
        Some(other) => bail!("{} declares sliding_window {other}, which is neither a window nor a list", path.display()),
    }
    Ok(())
}

/// Mistral Large 3's `moe` block read as DeepSeek's keys.
fn experts(config: &mut Map<String, Value>, moe: Map<String, Value>) {
    let keys = [
        ("route_every_n", "moe_layer_freq"),
        ("first_k_dense_replace", "first_k_dense_replace"),
        ("num_experts_per_tok", "num_experts_per_tok"),
        ("num_experts", "n_routed_experts"),
        ("expert_hidden_dim", "moe_intermediate_size"),
        ("routed_scale", "routed_scaling_factor"),
        ("num_shared_experts", "n_shared_experts"),
        ("num_expert_groups", "n_group"),
        ("num_expert_groups_per_tok", "topk_group"),
    ];
    for (from, to) in keys {
        if let Some(value) = moe.get(from) {
            config.insert(to.to_owned(), value.clone());
        }
    }
    config.insert("topk_method".to_owned(), Value::from("greedy"));
    config.insert("norm_topk_prob".to_owned(), Value::Bool(true));
    config.insert("scoring_func".to_owned(), Value::from("softmax"));
}

/// Mistral's `yarn` block as Transformers' YaRN `rope_parameters`: `beta`
/// and `alpha` are `beta_fast` and `beta_slow`, and `apply_scale: false`
/// leaves the magnitude uncorrected.
fn rope_yarn(config: &mut Map<String, Value>, yarn: &Value, path: &Path) -> Result<()> {
    let stated = |key: &str| yarn.get(key).and_then(Value::as_f64);
    let factor = stated("factor").with_context(|| format!("{} declares yarn without a factor", path.display()))?;
    let mut rope = Map::new();
    rope.insert("rope_type".to_owned(), Value::from("yarn"));
    rope.insert("mscale_all_dim".to_owned(), Value::from(MISTRAL_YARN_MSCALE_ALL_DIM));
    rope.insert("factor".to_owned(), Value::from(factor));
    if let Some(theta) = config.remove("rope_theta") {
        rope.insert("rope_theta".to_owned(), theta);
    }
    if let Some(original) = yarn.get("original_max_position_embeddings") {
        rope.insert("original_max_position_embeddings".to_owned(), original.clone());
    }
    if let Some(fast) = stated("beta") {
        rope.insert("beta_fast".to_owned(), Value::from(fast));
    }
    if let Some(slow) = stated("alpha") {
        rope.insert("beta_slow".to_owned(), Value::from(slow));
    }
    if yarn.get("apply_scale").and_then(Value::as_bool) == Some(false) {
        rope.insert("attention_factor".to_owned(), Value::from(UNSCALED_ATTENTION));
    }
    config.insert("rope_parameters".to_owned(), Value::Object(rope));
    Ok(())
}

/// The Mistral-format name of a tensor asked for by its Transformers name;
/// a name with no Mistral counterpart is returned as it is.
fn stored_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "tok_embeddings.weight".to_owned(),
        "model.norm.weight" => return "norm.weight".to_owned(),
        "lm_head.weight" => return "output.weight".to_owned(),
        _ => {}
    }
    let Some((layer, inner)) = name.strip_prefix("model.layers.").and_then(|rest| rest.split_once('.')) else {
        return name.to_owned();
    };
    let Some((module, leaf)) = inner.rsplit_once('.') else {
        return name.to_owned();
    };
    let renamed = match module.split_once('.') {
        None => match module {
            "input_layernorm" => Some("attention_norm".to_owned()),
            "post_attention_layernorm" => Some("ffn_norm".to_owned()),
            _ => None,
        },
        Some(("self_attn", projection)) => attention_name(projection).map(|stored| format!("attention.{stored}")),
        Some(("mlp", rest)) => feed_forward_name(rest),
        Some(_) => None,
    };
    match renamed {
        Some(module) => format!("layers.{layer}.{module}.{leaf}"),
        None => name.to_owned(),
    }
}

fn attention_name(projection: &str) -> Option<&'static str> {
    Some(match projection {
        "q_proj" => "wq",
        "k_proj" => "wk",
        "v_proj" => "wv",
        "o_proj" => "wo",
        "q_a_proj" => "wq_a",
        "q_a_layernorm" => "q_a_norm",
        "q_b_proj" => "wq_b",
        "kv_a_proj_with_mqa" => "wkv_a_with_mqa",
        "kv_a_layernorm" => "kv_a_norm",
        "kv_b_proj" => "wkv_b",
        _ => return None,
    })
}

fn projection_name(projection: &str) -> Option<&'static str> {
    Some(match projection {
        "gate_proj" => "w1",
        "down_proj" => "w2",
        "up_proj" => "w3",
        _ => return None,
    })
}

/// A dense feed-forward sits under `feed_forward`; a routed layer's router,
/// shared experts and experts sit beside the attention.
fn feed_forward_name(rest: &str) -> Option<String> {
    if rest == "gate" {
        return Some("gate".to_owned());
    }
    if let Some(projection) = rest.strip_prefix("shared_experts.") {
        return projection_name(projection).map(|stored| format!("shared_experts.{stored}"));
    }
    if let Some(expert) = rest.strip_prefix("experts.") {
        let (index, projection) = expert.split_once('.')?;
        return projection_name(projection).map(|stored| format!("experts.{index}.{stored}"));
    }
    projection_name(rest).map(|stored| format!("feed_forward.{stored}"))
}

/// The end-of-sequence token a Mistral-format config leaves out: the
/// tokenizer config's `eos_token` looked up among the added tokens.
pub fn end_of_sequence(tokenizer_config: Option<&Path>, tokenizer: &Path) -> Option<u64> {
    let read = |path: &Path| -> Option<Value> { serde_json::from_slice(&std::fs::read(path).ok()?).ok() };
    let text = read(tokenizer_config?)?.get("eos_token").and_then(|token| match token {
        Value::String(text) => Some(text.clone()),
        Value::Object(object) => object.get("content").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    })?;
    read(tokenizer)?
        .get("added_tokens")?
        .as_array()?
        .iter()
        .find(|token| token.get("content").and_then(Value::as_str) == Some(text.as_str()))?
        .get("id")?
        .as_u64()
}
