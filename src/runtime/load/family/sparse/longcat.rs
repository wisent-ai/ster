//! LongCat-Flash and its n-gram embeddings.

use std::path::Path;

use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;
use {crate::failures::model::Stated, anyhow::Result};

use crate::model::{Architecture, ExpertLayout, NgramSpec};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::{flag, number, text, whole, whole_or};

/// LongCat-Flash-Lite's n-gram embeddings (`longcat_flash_ngram`): tables
/// for orders 2 to `emb_neighbor_num`, `emb_split_num` of each, sized by
/// `ngram_vocab_size_ratio`, their runs restarting after `eos_token_id`
/// (the first, when it is a list; 2, the config class's default, when it
/// is left out).
pub(in crate::runtime::load::family) fn longcat_ngram(
    raw: &Value,
    hidden: usize,
    path: &Path,
) -> Result<NgramSpec> {
    let (Some(ratio), Some(splits), Some(neighbors)) = (
        whole(raw, "ngram_vocab_size_ratio", path)?,
        whole(raw, "emb_split_num", path)?,
        whole(raw, "emb_neighbor_num", path)?,
    ) else {
        refused!(
            Incomplete,
            in raw,
            ["ngram_vocab_size_ratio", "emb_split_num", "emb_neighbor_num"],
            "{} declares a LongCat n-gram model without ngram_vocab_size_ratio, emb_split_num or emb_neighbor_num",
            path.display()
        );
    };
    if splits == 0 || neighbors < 2 {
        refused!(
            OutOfRange,
            in raw,
            ["emb_split_num", "emb_neighbor_num"],
            "{} declares {splits} n-gram splits over neighbourhoods of {neighbors}; Ster needs at least one split and a neighbourhood of two tokens",
            path.display()
        );
    }
    let tables = splits * (neighbors - 1);
    if hidden % tables != 0 {
        refused!(
            Contradictory,
            in raw,
            ["hidden_size", "emb_split_num", "emb_neighbor_num"],
            "{} declares {tables} n-gram tables (emb_split_num times emb_neighbor_num - 1), which do not split a hidden width of {hidden} evenly",
            path.display()
        );
    }
    let eos = match raw.get("eos_token_id") {
        None | Some(Value::Null) => LONGCAT_NGRAM_EOS,
        Some(Value::Array(ids)) => match ids.first() {
            None => LONGCAT_NGRAM_EOS,
            Some(first) => first.as_u64().or_malformed(raw, &["eos_token_id"], || {
                format!(
                    "{} lists an eos_token_id that is not a token",
                    path.display()
                )
            })?,
        },
        Some(id) => id.as_u64().or_malformed(raw, &["eos_token_id"], || {
            format!(
                "{} declares an eos_token_id that is not a token",
                path.display()
            )
        })?,
    };
    Ok(NgramSpec {
        ratio,
        splits,
        neighbors,
        eos: eos as u32,
    })
}

/// The end-of-sequence token LongCat's n-gram config class assumes when a
/// config leaves `eos_token_id` out (`configuration_longcat_ngram.py`,
/// `eos_token_id=2`).
pub(in crate::runtime::load::family) const LONGCAT_NGRAM_EOS: u64 = 2;

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
pub(in crate::runtime::load::family) fn longcat_flash(
    raw: &Value,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(latent) = architecture.latent else {
        refused!(
            Incomplete,
            in raw,
            ["kv_lora_rank"],
            "{} declares a LongCat-Flash model without kv_lora_rank",
            path.display()
        );
    };
    if let Some(method) = text(raw, "attention_method", path)?.filter(|method| *method != "MLA") {
        refused!(
            Unimplemented,
            in raw,
            ["attention_method"],
            "{} declares attention_method {method:?}; Ster implements LongCat-Flash's MLA",
            path.display()
        );
    }
    let identity = whole_or(raw, "zero_expert_num", 0, path)?;
    if identity > 0 {
        if let Some(kind) = text(raw, "zero_expert_type", path)?.filter(|kind| *kind != "identity")
        {
            refused!(
                Unimplemented,
                in raw,
                ["zero_expert_num", "zero_expert_type"],
                "{} declares zero_expert_type {kind:?}; Ster implements identity zero experts",
                path.display()
            );
        }
    }
    architecture.interleaved_rotary = true;
    architecture.query_key_value_bias = flag(raw, "attention_bias", path)?;
    let hidden = llama.hidden_size as f64;
    // A scale a config switches on and whose rank it states; a switch that
    // cannot be read refuses the load instead of reading as off.
    let scale = |key: &str, rank: Option<usize>| -> Result<f64> {
        Ok(match rank {
            Some(rank) if flag(raw, key, path)? && rank > 0 => (hidden / rank as f64).sqrt(),
            _ => 1.0,
        })
    };
    architecture.latent_scales = Some((
        scale("mla_scale_q_lora", latent.query_rank)?,
        scale("mla_scale_kv_lora", Some(latent.key_value_rank))?,
    ));
    let mut routed = experts(
        raw,
        "n_routed_experts",
        "expert_ffn_hidden_size",
        flag(raw, "norm_topk_prob", path)?,
        ExpertLayout::LongCat,
        0,
        path,
    )?;
    routed.identity_experts = identity;
    routed.selection_bias = Some("mlp.router.e_score_correction_bias");
    routed.routed_scale = number(raw, "routed_scaling_factor", path)?;
    architecture.shortcut_experts = Some(routed);
    Ok(())
}
