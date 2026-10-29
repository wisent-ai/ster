//! LongCat-Flash and its n-gram embeddings.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, ExpertLayout, NgramSpec};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::{flag, number, text, whole};

/// LongCat-Flash-Lite's n-gram embeddings (`longcat_flash_ngram`): tables
/// for orders 2 to `emb_neighbor_num`, `emb_split_num` of each, sized by
/// `ngram_vocab_size_ratio`, their runs restarting after `eos_token_id`
/// (the first, when it is a list; 2, the config class's default, when it
/// is left out).
pub(in crate::runtime::load::family) fn longcat_ngram(
    raw: &Value,
    path: &Path,
) -> Result<NgramSpec> {
    let (Some(ratio), Some(splits), Some(neighbors)) = (
        whole(raw, "ngram_vocab_size_ratio"),
        whole(raw, "emb_split_num"),
        whole(raw, "emb_neighbor_num"),
    ) else {
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
        Some(Value::Array(ids)) => ids
            .first()
            .and_then(Value::as_u64)
            .unwrap_or(LONGCAT_NGRAM_EOS),
        Some(id) => id.as_u64().with_context(|| {
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
        bail!(
            "{} declares a LongCat-Flash model without kv_lora_rank",
            path.display()
        );
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
