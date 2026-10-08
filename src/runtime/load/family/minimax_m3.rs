//! MiniMax-M3's text decoder (`minimax_m3_vl_text`: the `text_config` of
//! `minimax_m3_vl`, whose weights sit below `language_model`), read as
//! vLLM's `MiniMaxM3SparseForCausalLM` and Transformers'
//! `MiniMaxM3VLTextModel` run it.
//!
//! Every norm scales by its weight plus one (`use_gemma_norm`), the query
//! and key are normed per head and rotate over `partial_rotary_factor` of
//! each head, and every feed-forward — the dense layers `moe_layer_freq`
//! marks zero, the shared expert and the routed experts — is SwiGLU-OAI
//! (`hidden_act` `swigluoai`) clamped by `swiglu_limit` with the gate's
//! sharpness `swiglu_alpha`. The routed layers score `num_local_experts`
//! Mixtral-named experts by sigmoid, choose with
//! `block_sparse_moe.e_score_correction_bias` added, renormalise and scale
//! by `routed_scaling_factor`, beside `n_shared_experts` shared experts of
//! `shared_intermediate_size` each. The layers `sparse_attention_config`
//! marks run MiniMax Sparse Attention: a block indexer chooses the key blocks
//! each query attends to.

use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use super::{every_layer, experts, flag, flagged_layers, number, text, whole};
use crate::model::{
    Architecture, BlockIndexSpec, ExpertLayout, QueryKeyNorm, Scoring, SharedExpert, SharedForm,
    SwigluLimit,
};

/// Where [`fill_keys`] keeps the `hidden_act` the config states, which it
/// replaces with the pointwise `silu` Transformers' `MiniMaxM3VLTextConfig`
/// writes there: SwiGLU-OAI is computed from `swiglu_alpha` and
/// `swiglu_limit`, never from the name.
const STATED_ACTIVATION: &str = "minimax_m3_hidden_act";

/// The Llama keys of a MiniMax-M3 config: its `intermediate_size` is each
/// routed expert's width and `dense_intermediate_size` its dense layers',
/// which becomes the Llama key.
pub(super) fn fill_keys(raw: &mut Value) {
    let experts = raw.get("intermediate_size").cloned();
    let dense = raw
        .get("dense_intermediate_size")
        .filter(|width| !width.is_null())
        .cloned();
    let activation = raw.get("hidden_act").cloned();
    let Some(object) = raw.as_object_mut() else {
        return;
    };
    if let Some(experts) = experts {
        object.entry("moe_intermediate_size").or_insert(experts);
    }
    if let Some(dense) = dense {
        object.insert("intermediate_size".to_owned(), dense);
    }
    if let Some(activation) = activation {
        object.insert(STATED_ACTIVATION.to_owned(), activation);
        object.insert("hidden_act".to_owned(), Value::from("silu"));
    }
}

pub(super) fn family(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    match text(raw, STATED_ACTIVATION) {
        Some("swigluoai") => {}
        other => bail!(
            "{} declares hidden_act {other:?}; MiniMax-M3's feed-forwards are swigluoai",
            path.display()
        ),
    }
    if flag(raw, "attention_output_gate") {
        bail!(
            "{} declares attention_output_gate; Ster implements MiniMax-M3 without the attention output gate, as vLLM and Transformers do",
            path.display()
        );
    }
    if raw.get("use_gemma_norm").and_then(Value::as_bool) == Some(false) {
        bail!(
            "{} declares use_gemma_norm false; MiniMax-M3's norms scale by their weight plus one",
            path.display()
        );
    }
    architecture.norm_offset = true;
    if raw.get("use_qk_norm").and_then(Value::as_bool) == Some(false) {
        bail!(
            "{} declares use_qk_norm false; MiniMax-M3 norms every query and key head",
            path.display()
        );
    }
    match text(raw, "qk_norm_type") {
        None | Some("per_head") => architecture.query_key_norm = QueryKeyNorm::PerHead,
        Some(other) => bail!(
            "{} declares qk_norm_type {other:?}; MiniMax-M3 norms per head",
            path.display()
        ),
    }
    let (Some(limit), Some(alpha)) = (number(raw, "swiglu_limit"), number(raw, "swiglu_alpha"))
    else {
        bail!(
            "{} declares swigluoai without swiglu_limit and swiglu_alpha",
            path.display()
        );
    };
    let clamp = SwigluLimit::Oai { limit, alpha };
    let every = every_layer(layers, path)?;
    let routed_layers = match raw.get("moe_layer_freq") {
        Some(_) => flagged_layers(raw, "moe_layer_freq", layers, path)?,
        None => every,
    };
    if routed_layers != every && whole(raw, "dense_intermediate_size").is_none() {
        bail!(
            "{} marks dense layers in moe_layer_freq without dense_intermediate_size",
            path.display()
        );
    }
    architecture.dense_swiglu_limit = Some(clamp.clone());
    let mut routed = experts(
        raw,
        "num_local_experts",
        "moe_intermediate_size",
        true,
        ExpertLayout::Mixtral,
        every & !routed_layers,
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
    routed.routed_scale = number(raw, "routed_scaling_factor");
    routed.swiglu_limit = Some(clamp);
    routed.shared = match whole(raw, "n_shared_experts").and_then(NonZeroUsize::new) {
        Some(shared) => {
            let Some(width) = whole(raw, "shared_intermediate_size") else {
                bail!(
                    "{} declares n_shared_experts without shared_intermediate_size",
                    path.display()
                );
            };
            Some(SharedExpert {
                intermediate: shared.get() * width,
                module: "block_sparse_moe.shared_experts",
                gated: false,
                form: SharedForm::GateUpDown,
            })
        }
        None => None,
    };
    architecture.experts = Some(routed);
    architecture.block_index = block_index(raw, layers, path)?;
    Ok(())
}

/// MiniMax Sparse Attention's block indexer from `sparse_attention_config`,
/// or `None` when the config runs every layer densely.
fn block_index(raw: &Value, layers: usize, path: &Path) -> Result<Option<BlockIndexSpec>> {
    let Some(sparse) = raw
        .get("sparse_attention_config")
        .filter(|sparse| sparse.is_object())
    else {
        return Ok(None);
    };
    if sparse.get("use_sparse_attention").and_then(Value::as_bool) == Some(false) {
        return Ok(None);
    }
    let indexed = flagged_layers(sparse, "sparse_attention_freq", layers, path)?;
    match text(sparse, "sparse_score_type") {
        None | Some("max") => {}
        Some(other) => bail!(
            "{} declares sparse_score_type {other:?}; Ster scores a key block by its best key (max)",
            path.display()
        ),
    }
    if whole(sparse, "sparse_init_block")
        .and_then(NonZeroUsize::new)
        .is_some()
    {
        bail!(
            "{} keeps sparse_init_block leading blocks always visible; Ster implements the local blocks Transformers' MiniMax-M3 indexer keeps, not leading ones",
            path.display()
        );
    }
    // Every sparse layer of MiniMax-M3 disables the indexer's own value and
    // output branch; neither vLLM nor Transformers implements it.
    if sparse.get("sparse_disable_index_value").is_some() {
        let disabled = flagged_layers(sparse, "sparse_disable_index_value", layers, path)?;
        if indexed & disabled != indexed {
            bail!(
                "{} enables the indexer's value branch on a sparse layer (sparse_disable_index_value); Ster implements MiniMax-M3's indexer without it",
                path.display()
            );
        }
    }
    let size = |key: &str| -> Result<usize> {
        match whole(sparse, key) {
            Some(size) => Ok(size),
            None => bail!(
                "{} declares sparse_attention_config without {key}",
                path.display()
            ),
        }
    };
    let heads = size("sparse_num_index_heads")?;
    if Some(heads) != whole(raw, "num_key_value_heads") {
        bail!(
            "{} declares {heads} index heads (sparse_num_index_heads) and a different num_key_value_heads; MiniMax-M3 chooses one block selection per key-value head",
            path.display()
        );
    }
    let (Some(block), Some(top_blocks)) = (
        NonZeroUsize::new(size("sparse_block_size")?),
        NonZeroUsize::new(size("sparse_topk_blocks")?),
    ) else {
        bail!(
            "{} declares an empty sparse_block_size or sparse_topk_blocks",
            path.display()
        );
    };
    let local_blocks = size("sparse_local_block")?;
    if local_blocks > top_blocks.get() {
        bail!(
            "{} keeps {local_blocks} local key blocks (sparse_local_block) among only {top_blocks} kept ones (sparse_topk_blocks)",
            path.display()
        );
    }
    Ok(Some(BlockIndexSpec {
        heads,
        head_dim: size("sparse_index_dim")?,
        block,
        top_blocks: top_blocks.get(),
        local_blocks,
        layers: indexed,
    }))
}
