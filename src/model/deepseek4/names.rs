//! DeepSeek-V4's own tensor names, which DeepSeek's release and checkpoints
//! written by its reference code keep, answering each Transformers name Ster
//! asks for (the inverse of Transformers' `deepseek_v4` conversion mapping).

/// How a checkpoint in DeepSeek's own layout spells what Ster reads.
#[derive(Debug, Clone)]
pub(crate) struct NativeNames {
    /// What the native names sit below: nothing in DeepSeek's release,
    /// `model.` in checkpoints that keep the module root.
    pub root: String,
    /// The main attention's key-value norm: `kv_norm`, or `norm` in older
    /// writers.
    pub key_value_norm: &'static str,
}

impl NativeNames {
    /// The stored name of the Transformers name `name`.
    pub(crate) fn stored(&self, name: &str) -> String {
        if name == "lm_head.weight" {
            return "head.weight".to_owned();
        }
        let Some(rest) = name.strip_prefix("model.") else {
            return name.to_owned();
        };
        let mapped = match rest {
            "embed_tokens.weight" => "embed.weight".to_owned(),
            "hc_head.hc_fn" => "hc_head_fn".to_owned(),
            "hc_head.hc_base" => "hc_head_base".to_owned(),
            "hc_head.hc_scale" => "hc_head_scale".to_owned(),
            _ => match rest
                .strip_prefix("layers.")
                .and_then(|tail| tail.split_once('.'))
            {
                Some((layer, tail)) => format!("layers.{layer}.{}", self.layer(tail)),
                None => rest.to_owned(),
            },
        };
        format!("{}{mapped}", self.root)
    }

    fn layer(&self, tail: &str) -> String {
        let renamed = [
            ("input_layernorm.", "attn_norm."),
            ("post_attention_layernorm.", "ffn_norm."),
        ];
        for (from, to) in renamed {
            if let Some(rest) = tail.strip_prefix(from) {
                return format!("{to}{rest}");
            }
        }
        if let Some(rest) = tail.strip_prefix("self_attn.") {
            return format!("attn.{}", self.attention(rest));
        }
        if let Some(rest) = tail.strip_prefix("mlp.") {
            return format!("ffn.{}", feed_forward(rest));
        }
        tail.to_owned()
    }

    fn attention(&self, tail: &str) -> String {
        if let Some(rest) = tail.strip_prefix("compressor.indexer.") {
            return match rest.split_once('.') {
                Some(("q_b_proj", leaf)) => format!("indexer.wq_b.{leaf}"),
                Some(("scorer", leaf)) => format!("indexer.{leaf}"),
                _ => format!("indexer.compressor.{}", compressor(rest)),
            };
        }
        if let Some(rest) = tail.strip_prefix("compressor.") {
            return format!("compressor.{}", compressor(rest));
        }
        let (module, leaf) = tail.split_once('.').unwrap_or((tail, ""));
        let module = match module {
            "q_a_proj" => "wq_a",
            "q_a_norm" => "q_norm",
            "q_b_proj" => "wq_b",
            "kv_proj" => "wkv",
            "kv_norm" => self.key_value_norm,
            "o_a_proj" => "wo_a",
            "o_b_proj" => "wo_b",
            "sinks" => "attn_sink",
            other => other,
        };
        if leaf.is_empty() {
            module.to_owned()
        } else {
            format!("{module}.{leaf}")
        }
    }
}

fn compressor(tail: &str) -> String {
    let (module, leaf) = tail.split_once('.').unwrap_or((tail, ""));
    let module = match module {
        "kv_proj" => "wkv",
        "gate_proj" => "wgate",
        "position_bias" => "ape",
        "kv_norm" => "norm",
        other => other,
    };
    if leaf.is_empty() {
        module.to_owned()
    } else {
        format!("{module}.{leaf}")
    }
}

fn feed_forward(tail: &str) -> String {
    let renamed = [
        ("gate.e_score_correction_bias", "gate.bias"),
        ("shared_experts.gate_proj.", "shared_experts.w1."),
        ("shared_experts.up_proj.", "shared_experts.w3."),
        ("shared_experts.down_proj.", "shared_experts.w2."),
    ];
    for (from, to) in renamed {
        if let Some(rest) = tail.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    tail.to_owned()
}
