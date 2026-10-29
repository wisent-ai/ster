//! Ling 3.0's Kimi Delta Attention.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, DeltaRuleForm, DeltaRuleSpec, GateFunction};
use crate::runtime::load::family::read::{flag, number, text, whole};

/// Ling 3.0's linear layers: Kimi Delta Attention under `attention`, with
/// full-rank `f_proj` (the decay input, one per key channel) and `g_proj`
/// (the output gate) where Kimi factors them, `num_attention_heads` heads of
/// `head_dim` for query, key and value, and `short_conv_kernel_size` taps.
/// The decay is `kda_lower_bound · sigmoid(exp(A_log) · (f + dt_bias))`
/// under `kda_safe_gate` (vLLM's `bailing_moe_v3.py` and FLA's
/// `naive_kda_lowerbound_gate`), Kimi's `-exp(A_log) · softplus(f +
/// dt_bias)` otherwise. A layer is linear unless it closes a
/// `layer_group_size` group or follows the last whole group
/// (`_is_kda_layer`). The latent-attention layers multiply each head's
/// output by the sigmoid of its `attention.g_proj` logit under
/// `gated_attention_proj_granularity_type` `head_wise`.
pub(in crate::runtime::load::family) fn ling_kda(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    if raw.get("no_kda_lora").and_then(Value::as_bool) == Some(false) || flag(raw, "use_kda_lora") {
        bail!(
            "{} factors its Kimi Delta Attention projections (no_kda_lora false or use_kda_lora); Ster implements Ling 3.0's full-rank f_proj and g_proj",
            path.display()
        );
    }
    for key in ["use_nGPT", "value_norm", "up_proj_norm", "use_mla_nope"] {
        if flag(raw, key) {
            bail!(
                "{} declares {key}; Ster implements Ling 3.0 without it",
                path.display()
            );
        }
    }
    let Some(head_dim) = whole(raw, "head_dim") else {
        bail!(
            "{} declares a Ling 3.0 model without head_dim",
            path.display()
        );
    };
    let Some(kernel) = whole(raw, "short_conv_kernel_size").filter(|kernel| *kernel > 0) else {
        bail!(
            "{} declares a Ling 3.0 model without short_conv_kernel_size",
            path.display()
        );
    };
    let group = whole(raw, "layer_group_size").unwrap_or(1).max(1);
    let whole_groups = layers / group * group;
    let linear = (0..layers)
        .filter(|layer| (layer + 1) % group != 0 && *layer < whole_groups)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let heads = llama.num_attention_heads;
    let decay_floor = if raw
        .get("kda_safe_gate")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        Some(number(raw, "kda_lower_bound").unwrap_or(KDA_LOWER_BOUND))
    } else {
        None
    };
    architecture.delta_rule = Some(DeltaRuleSpec {
        key_heads: heads,
        value_heads: heads,
        key_dim: head_dim,
        value_dim: head_dim,
        kernel,
        layers: linear,
        form: DeltaRuleForm::Ling,
        negative_eigenvalues: false,
        decay_floor,
        full_rank_gate: true,
    });
    architecture.interleaved_rotary = raw
        .get("rope_interleave")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    architecture.head_gate = match text(raw, "gated_attention_proj_granularity_type") {
        None => None,
        Some("head_wise") => Some(GateFunction::Sigmoid),
        Some(other) => bail!(
            "{} declares gated_attention_proj_granularity_type {other:?}; Ster implements Ling 3.0's head_wise gate",
            path.display()
        ),
    };
    Ok(())
}

/// The floor Kimi Delta Attention's safe gate puts under each log-decay when
/// a config leaves `kda_lower_bound` out (vLLM's `bailing_moe_v3.py`,
/// `getattr(config, "kda_lower_bound", -5.0)`).
pub(in crate::runtime::load::family) const KDA_LOWER_BOUND: f64 = -5.0;
