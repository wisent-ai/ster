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

use anyhow::Result;
use serde_json::Value;

mod blocks;

use crate::model::{
    Architecture, DenseClamp, ExpertLayout, QueryKeyNorm, Scoring, SharedExpert, SharedForm,
    SwigluLimit,
};
use crate::runtime::load::family::experts::experts;
use crate::runtime::load::family::read::layers::{every_layer, flagged_layers};
use crate::runtime::load::family::read::{flag, flag_or, number, text, whole};
use blocks::{block_index, index_fits};

/// Where [`fill_keys`] keeps the `hidden_act` the config states, which it
/// replaces with the pointwise `silu` Transformers' `MiniMaxM3VLTextConfig`
/// writes there: SwiGLU-OAI is computed from `swiglu_alpha` and
/// `swiglu_limit`, never from the name.
const STATED_ACTIVATION: &str = "minimax_m3_hidden_act";

/// The Llama keys of a MiniMax-M3 config: its `intermediate_size` is each
/// routed expert's width and `dense_intermediate_size` its dense layers',
/// which becomes the Llama key.
pub(in crate::runtime::load::family) fn fill_keys(raw: &mut Value, path: &Path) -> Result<()> {
    let experts = raw.get("intermediate_size").cloned();
    let dense = whole(raw, "dense_intermediate_size", path)?;
    let activation = raw.get("hidden_act").cloned();
    let Some(object) = raw.as_object_mut() else {
        return Ok(());
    };
    if let Some(experts) = experts {
        object.entry("moe_intermediate_size").or_insert(experts);
    }
    if let Some(dense) = dense {
        object.insert("intermediate_size".to_owned(), Value::from(dense));
    }
    if let Some(activation) = activation {
        object.insert(STATED_ACTIVATION.to_owned(), activation);
        object.insert("hidden_act".to_owned(), Value::from("silu"));
    }
    Ok(())
}

/// `query_heads` is the Llama reading's `num_attention_heads`.
pub(in crate::runtime::load::family) fn family(
    raw: &Value,
    layers: usize,
    query_heads: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    match text(raw, STATED_ACTIVATION, path)? {
        Some("swigluoai") => {}
        other => refused!(
            Unimplemented,
            in raw,
            [STATED_ACTIVATION],
            "{} declares hidden_act {other:?}; MiniMax-M3's feed-forwards are swigluoai",
            path.display()
        ),
    }
    if flag(raw, "attention_output_gate", path)? {
        refused!(
            Unimplemented,
            in raw,
            ["attention_output_gate"],
            "{} declares attention_output_gate; Ster implements MiniMax-M3 without the attention output gate, as vLLM and Transformers do",
            path.display()
        );
    }
    if !flag_or(raw, "use_gemma_norm", true, path)? {
        refused!(
            Unimplemented,
            in raw,
            ["use_gemma_norm"],
            "{} declares use_gemma_norm false; MiniMax-M3's norms scale by their weight plus one",
            path.display()
        );
    }
    architecture.norm_offset = true;
    if !flag_or(raw, "use_qk_norm", true, path)? {
        refused!(
            Unimplemented,
            in raw,
            ["use_qk_norm"],
            "{} declares use_qk_norm false; MiniMax-M3 norms every query and key head",
            path.display()
        );
    }
    match text(raw, "qk_norm_type", path)? {
        None | Some("per_head") => architecture.query_key_norm = QueryKeyNorm::PerHead,
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["qk_norm_type"],
            "{} declares qk_norm_type {other:?}; MiniMax-M3 norms per head",
            path.display()
        ),
    }
    let (Some(limit), Some(alpha)) = (number(raw, "swiglu_limit", path)?, number(raw, "swiglu_alpha", path)?)
    else {
        refused!(
            Incomplete,
            in raw,
            ["swiglu_limit", "swiglu_alpha"],
            "{} declares swigluoai without swiglu_limit and swiglu_alpha",
            path.display()
        );
    };
    let clamp = SwigluLimit::Oai { limit, alpha };
    let every = every_layer(layers, path)?;
    let routed_layers = match raw.get("moe_layer_freq") {
        None | Some(Value::Null) => every,
        Some(_) => flagged_layers(raw, "moe_layer_freq", layers, path)?,
    };
    if routed_layers != every && whole(raw, "dense_intermediate_size", path)?.is_none() {
        refused!(
            Incomplete,
            in raw,
            ["moe_layer_freq", "dense_intermediate_size"],
            "{} marks dense layers in moe_layer_freq without dense_intermediate_size",
            path.display()
        );
    }
    architecture.dense_swiglu_limit = Some(DenseClamp::Oai { limit, alpha });
    let mut routed = experts(
        raw,
        "num_local_experts",
        "moe_intermediate_size",
        true,
        ExpertLayout::Mixtral,
        every & !routed_layers,
        path,
    )?;
    routed.scoring = match text(raw, "scoring_func", path)? {
        None | Some("sigmoid") => Scoring::Sigmoid,
        Some("softmax") => Scoring::Softmax,
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["scoring_func"],
            "{} declares scoring_func {other:?}; Ster implements softmax and sigmoid expert scores",
            path.display()
        ),
    };
    routed.selection_bias = flag_or(raw, "use_routing_bias", true, path)?
        .then_some("block_sparse_moe.e_score_correction_bias");
    routed.routed_scale = number(raw, "routed_scaling_factor", path)?;
    routed.swiglu_limit = Some(clamp);
    routed.shared = match whole(raw, "n_shared_experts", path)?.and_then(NonZeroUsize::new) {
        Some(shared) => {
            let Some(width) = whole(raw, "shared_intermediate_size", path)? else {
                refused!(
                    Incomplete,
                    in raw,
                    ["n_shared_experts", "shared_intermediate_size"],
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
    if let Some(index) = architecture.block_index {
        index_fits(raw, index, query_heads, architecture.rotary_dim, path)?;
    }
    Ok(())
}
