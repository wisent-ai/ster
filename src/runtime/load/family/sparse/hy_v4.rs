//! HY-V4 and the hyper-connections DeepSeek-V4, GLM-5-Next and HY-V4 share.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::model::{
    Activation, Architecture, ExpertLayout, GateFunction, HyperConnections, HyperForm, Scoring,
    SharedExpert, SharedForm, StoredNames, SwigluLimit,
};
use crate::runtime::load::family::experts::index::sparse_index;
use crate::runtime::load::family::experts::{experts, qwen_dense_layers};
use crate::runtime::load::family::read::layers::{every_layer, fits};
use crate::runtime::load::family::read::{flag, number, text, whole};

/// How many leading layers keep a dense feed-forward when an HY-V4 config
/// lists no `mlp_layer_types` (`HYV4Config.__post_init__`,
/// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_DENSE_LAYERS: usize = 1;

/// HY-V4's `hc_mult` when its config leaves it out (`HYV4Config`,
/// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_STREAMS: usize = 4;

/// HY-V4's `hc_magnitude` when its config leaves it out (`HYV4Config`,
/// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_MAGNITUDE: f64 = 2.0;

/// HY-V4's `hc_eps` when its config leaves it out (`HYV4Config`,
/// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_HC_EPS: f64 = 1e-6;

/// HY-V4's `n_shared_experts` when its config leaves it out (`HYV4Config`,
/// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_SHARED_EXPERTS: usize = 1;

/// HY-V4's `routed_scaling_factor` when its config leaves it out
/// (`HYV4Config`, https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_ROUTED_SCALE: f64 = 2.827;

/// HY-V4's `swiglu_limit` when its config leaves it out (`HYV4Config`,
/// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_SWIGLU_LIMIT: f64 = 10.0;

/// HY-V4 (`hy_v4`, Tencent's Hy4), read as vLLM's `vllm/models/hy_v4`
/// serves it: DeepSeek-V3's latent attention rotating adjacent pairs, with a
/// learned sink logit per head in every softmax (`learnable_sink`) and,
/// under `gated_mla`, its output multiplied elementwise by
/// `sigmoid(linear_gate(x))` before `o_proj`; DeepSeek Sparse Attention
/// whose `indexer_types` `shared` layers reuse the last indexed layer's
/// choice, its indexer rotating the last `qk_rope_head_dim` components of
/// each head; under `enable_ihc`, `hc_mult` independent hyper-connected
/// streams collapsed by `hc_head`; a dense feed-forward on the layers
/// `mlp_layer_types` marks `dense`, elsewhere experts scored by a sigmoid,
/// chosen with `mlp.gate.e_score_correction_bias`, renormalised under
/// `norm_topk_prob` and scaled by `routed_scaling_factor`, beside
/// `n_shared_experts` shared experts. The routed experts are SwiGLU clamped
/// by `swiglu_limit` (unclamped at zero); the dense and shared feed-forwards
/// run `hidden_act` unclamped. Under `soft_logits_capping` the logits are
/// soft-capped at `soft_logits_capping_logits`. A key the config class
/// states a default for reads as that default when the config leaves it
/// out. The release's tensor names are read through Transformers' `hy_v4`
/// renames ([`StoredNames::HyV4`]).
pub(in crate::runtime::load::family) fn hy_v4(
    raw: &Value,
    layers: usize,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    fits(layers, path)?;
    if architecture
        .latent
        .is_none_or(|latent| latent.query_rank.is_none())
    {
        bail!(
            "{} declares HY-V4 attention without kv_lora_rank and q_lora_rank; Ster implements its latent attention, whose indexer reads the latent query",
            path.display()
        );
    }
    // The switches `HYV4Config` turns on unless the config turns them off.
    let on = |key: &str| raw.get(key).and_then(Value::as_bool) != Some(false);
    architecture.stored_names = StoredNames::HyV4;
    // vLLM rotates adjacent pairs in the attention as in the indexer
    // (`is_neox_style=False`): the checkpoint is in Megatron's layout.
    architecture.interleaved_rotary = true;
    architecture.query_key_value_bias = flag(raw, "attention_bias");
    architecture.output_bias = architecture.query_key_value_bias;
    if on("learnable_sink") {
        architecture.attention_sinks = every_layer(layers, path)?;
    }
    if on("gated_mla") {
        match text(raw, "gating_type") {
            None | Some("elementwise") => {
                architecture.attention_gate = Some(GateFunction::Sigmoid);
            }
            other => bail!(
                "{} declares gating_type {other:?}; Ster implements HY-V4's elementwise output gate",
                path.display()
            ),
        }
    }
    architecture.sparse_index = Some(sparse_index(raw, "hy_v4", layers, path)?);
    if on("enable_ihc") {
        let streams = match whole(raw, "hc_mult") {
            Some(stated) => stated,
            None => HY_V4_STREAMS,
        };
        if std::num::NonZeroUsize::new(streams).is_none() {
            bail!(
                "{} declares hc_mult zero under enable_ihc; hyper-connections need at least one stream",
                path.display()
            );
        }
        architecture.hyper_connections = Some(HyperConnections {
            streams,
            form: HyperForm::Independent {
                magnitude: match number(raw, "hc_magnitude") {
                    Some(stated) => stated,
                    None => HY_V4_MAGNITUDE,
                },
            },
            eps: match number(raw, "hc_eps") {
                Some(stated) => stated,
                None => HY_V4_HC_EPS,
            },
            learned_head: true,
        });
    }
    if flag(raw, "soft_logits_capping") {
        let Some(cap) = number(raw, "soft_logits_capping_logits") else {
            bail!(
                "{} declares soft_logits_capping without soft_logits_capping_logits",
                path.display()
            );
        };
        architecture.final_softcap = Some(cap);
    }
    let dense_layers = match raw.get("mlp_layer_types") {
        Some(_) => qwen_dense_layers(raw, layers, path)?,
        None => every_layer(HY_V4_DENSE_LAYERS.min(layers), path)?,
    };
    let mut routed = experts(
        raw,
        "n_routed_experts",
        "moe_intermediate_size",
        on("norm_topk_prob"),
        ExpertLayout::Qwen,
        dense_layers,
        path,
    )?;
    routed.scoring = Scoring::Sigmoid;
    routed.selection_bias = Some("mlp.gate.e_score_correction_bias");
    routed.routed_scale = Some(match number(raw, "routed_scaling_factor") {
        Some(stated) => stated,
        None => HY_V4_ROUTED_SCALE,
    });
    let limit = match number(raw, "swiglu_limit") {
        Some(stated) => stated,
        None => HY_V4_SWIGLU_LIMIT,
    };
    // vLLM clamps the routed experts only under a positive limit.
    routed.swiglu_limit =
        (limit.is_normal() && limit.is_sign_positive()).then_some(SwigluLimit::RoutedInner(limit));
    routed.routed_activation = Some(Activation::Silu);
    let shared = match whole(raw, "n_shared_experts") {
        Some(stated) => stated,
        None => HY_V4_SHARED_EXPERTS,
    };
    routed.shared = std::num::NonZeroUsize::new(shared).map(|shared| SharedExpert {
        intermediate: shared.get() * routed.intermediate,
        module: "mlp.shared_experts",
        gated: false,
        form: SharedForm::GateUpDown,
    });
    architecture.experts = Some(routed);
    Ok(())
}

/// The hyper-connections DeepSeek-V4 and GLM-5-Next state: `hc_mult`
/// streams, `hc_sinkhorn_iters` rounds, `hc_eps`; `learned_head` when the
/// model collapses its last streams through `hc_head`.
pub(in crate::runtime::load::family) fn hyper_connections(
    raw: &Value,
    learned_head: bool,
    path: &Path,
) -> Result<HyperConnections> {
    let size = |key: &str| -> Result<usize> {
        whole(raw, key).filter(|size| *size > 0).with_context(|| {
            format!(
                "{} declares hyper-connections without {key}",
                path.display()
            )
        })
    };
    let Some(eps) = number(raw, "hc_eps") else {
        bail!(
            "{} declares hyper-connections without hc_eps",
            path.display()
        );
    };
    Ok(HyperConnections {
        streams: size("hc_mult")?,
        form: HyperForm::Manifold {
            sinkhorn_iterations: size("hc_sinkhorn_iters")?,
        },
        eps,
        learned_head,
    })
}
