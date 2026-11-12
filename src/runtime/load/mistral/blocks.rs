//! Mistral's nested `params.json` blocks put in Transformers' keys: the
//! `sliding_window` list, Mistral Large 3's `moe` block and the `yarn` block.

use std::path::Path;

use serde_json::{Map, Value};
use {crate::failures::model::Stated, anyhow::Result};

use crate::model::keys::{mistyped, within, within_or};

/// What `apply_scale: false` makes YaRN's magnitude correction: none, which
/// Transformers spells as an attention factor of one.
const UNSCALED_ATTENTION: f64 = 1.0;
/// The `mscale_all_dim` every Mistral YaRN rotation gets.
const MISTRAL_YARN_MSCALE_ALL_DIM: f64 = 1.0;

/// A `sliding_window` list (a window or null per layer, repeated over the
/// layers) becomes `layer_types` and one window; a single window covers
/// every layer.
pub(super) fn sliding_window(config: &mut Map<String, Value>, path: &Path) -> Result<()> {
    match config.get("sliding_window").cloned() {
        None | Some(Value::Null) | Some(Value::Number(_)) => {}
        Some(Value::Array(pattern)) => {
            let layers = match config.get("num_hidden_layers") {
                None | Some(Value::Null) => refused!(
                    Incomplete,
                    ["sliding_window" => Value::Array(pattern.clone()), "n_layers" => "absent"],
                    "{} lists a sliding_window pattern but states no n_layers to repeat it over; state n_layers",
                    path.display()
                ),
                Some(stated) => match stated.as_u64() {
                    Some(layers) => layers as usize,
                    None => return Err(mistyped("n_layers", stated, "a whole number", path)),
                },
            };
            if pattern.is_empty() || layers % pattern.len() != 0 {
                refused!(
                    Contradictory,
                    ["sliding_window" => Value::Array(pattern.clone()), "n_layers" => layers],
                    "{} repeats a sliding_window pattern of {} over {layers} layers unevenly",
                    path.display(),
                    pattern.len()
                );
            }
            let mut windows: Vec<u64> = Vec::new();
            for (layer, window) in pattern.iter().enumerate() {
                match window {
                    Value::Null => {}
                    stated => match stated.as_u64() {
                        Some(window) => windows.push(window),
                        None => {
                            let key = format!("sliding_window[{layer}]");
                            return Err(mistyped(&key, stated, "a whole number or null", path));
                        }
                    },
                }
            }
            if windows.iter().any(|window| Some(window) != windows.first()) {
                refused!(
                    Unimplemented,
                    ["sliding_window" => Value::Array(pattern.clone())],
                    "{} lists more than one sliding window; Ster gives every windowed layer one",
                    path.display()
                );
            }
            let kinds: Vec<Value> = (0..layers)
                .map(|layer| {
                    let kind = if pattern[layer % pattern.len()].is_null() {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    };
                    Value::from(kind)
                })
                .collect();
            config.insert("layer_types".to_owned(), Value::Array(kinds));
            // A pattern of nulls alone windows no layer, so it states no
            // window either.
            let window = match windows.first() {
                Some(window) => Value::from(*window),
                None => Value::Null,
            };
            config.insert("sliding_window".to_owned(), window);
        }
        Some(other) => refused!(
            Malformed,
            ["sliding_window" => other],
            "{} declares sliding_window {other}, which is neither a window nor a list",
            path.display()
        ),
    }
    Ok(())
}

/// Mistral Large 3's `moe` block read as DeepSeek's keys.
pub(super) fn experts(config: &mut Map<String, Value>, moe: Map<String, Value>) {
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
pub(super) fn rope_yarn(config: &mut Map<String, Value>, yarn: &Value, path: &Path) -> Result<()> {
    let stated = |key: &str| within::<f64>(yarn, "yarn", key, path);
    let factor = stated("factor")?.or_refuse(yarn, &["factor"], || {
        format!("{} declares yarn without a factor", path.display())
    })?;
    let mut rope = Map::new();
    rope.insert("rope_type".to_owned(), Value::from("yarn"));
    rope.insert(
        "mscale_all_dim".to_owned(),
        Value::from(MISTRAL_YARN_MSCALE_ALL_DIM),
    );
    rope.insert("factor".to_owned(), Value::from(factor));
    if let Some(theta) = config.remove("rope_theta") {
        rope.insert("rope_theta".to_owned(), theta);
    }
    if let Some(original) = yarn.get("original_max_position_embeddings") {
        rope.insert(
            "original_max_position_embeddings".to_owned(),
            original.clone(),
        );
    }
    if let Some(fast) = stated("beta")? {
        rope.insert("beta_fast".to_owned(), Value::from(fast));
    }
    if let Some(slow) = stated("alpha")? {
        rope.insert("beta_slow".to_owned(), Value::from(slow));
    }
    if !within_or(yarn, "yarn", "apply_scale", true, path)? {
        rope.insert(
            "attention_factor".to_owned(),
            Value::from(UNSCALED_ATTENTION),
        );
    }
    config.insert("rope_parameters".to_owned(), Value::Object(rope));
    Ok(())
}
