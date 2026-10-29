//! Mistral's own tensor names, answered for the Transformers names the
//! decoder asks for, following the weight mappers of vLLM's `llama.py` and
//! `mistral_large_3.py`.

/// The Mistral-format name of a tensor asked for by its Transformers name;
/// a name with no Mistral counterpart is returned as it is.
pub(super) fn stored_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "tok_embeddings.weight".to_owned(),
        "model.norm.weight" => return "norm.weight".to_owned(),
        "lm_head.weight" => return "output.weight".to_owned(),
        _ => {}
    }
    let Some((layer, inner)) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
    else {
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
        Some(("self_attn", projection)) => {
            attention_name(projection).map(|stored| format!("attention.{stored}"))
        }
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
