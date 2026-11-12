//! Ling 3.0's Kimi Delta Attention.

use std::path::Path;

use anyhow::Result;
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, DeltaRuleForm, DeltaRuleSpec, GateFunction};
use crate::runtime::load::family::read::{flag, flag_or, number_or, text, whole, whole_or};

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
    if !flag_or(raw, "no_kda_lora", true, path)? || flag(raw, "use_kda_lora", path)? {
        refused!(
            Unimplemented,
            in raw,
            ["no_kda_lora", "use_kda_lora"],
            "{} factors its Kimi Delta Attention projections (no_kda_lora false or use_kda_lora); Ster implements Ling 3.0's full-rank f_proj and g_proj",
            path.display()
        );
    }
    for key in ["use_nGPT", "value_norm", "up_proj_norm", "use_mla_nope"] {
        if flag(raw, key, path)? {
            refused!(
                Unimplemented,
                in raw,
                [key],
                "{} declares {key}; Ster implements Ling 3.0 without it",
                path.display()
            );
        }
    }
    let Some(head_dim) = whole(raw, "head_dim", path)? else {
        refused!(
            Incomplete,
            in raw,
            ["head_dim"],
            "{} declares a Ling 3.0 model without head_dim",
            path.display()
        );
    };
    let Some(kernel) = whole(raw, "short_conv_kernel_size", path)?.filter(|kernel| *kernel > 0) else {
        refused!(
            Incomplete,
            in raw,
            ["short_conv_kernel_size"],
            "{} declares a Ling 3.0 model without short_conv_kernel_size",
            path.display()
        );
    };
    let group = whole_or(raw, "layer_group_size", 1, path)?.max(1);
    let whole_groups = layers / group * group;
    let linear = (0..layers)
        .filter(|layer| (layer + 1) % group != 0 && *layer < whole_groups)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    let heads = llama.num_attention_heads;
    let decay_floor = if flag_or(raw, "kda_safe_gate", true, path)?
    {
        Some(number_or(raw, "kda_lower_bound", KDA_LOWER_BOUND, path)?)
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
    architecture.interleaved_rotary = flag_or(raw, "rope_interleave", true, path)?;
    architecture.head_gate = match text(raw, "gated_attention_proj_granularity_type", path)? {
        None => None,
        Some("head_wise") => Some(GateFunction::Sigmoid),
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["gated_attention_proj_granularity_type"],
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
