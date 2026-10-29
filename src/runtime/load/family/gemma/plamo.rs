//! PLaMo 3 and MuseGlimmer.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, GateFunction, Names, QkvLayout, QueryKeyNorm};
use crate::runtime::load::family::read::layers::{fits, listed_layers};
use crate::runtime::load::family::read::{flag, number, text, whole};
use crate::runtime::load::family::rope::bases::layer_bases;
use crate::runtime::load::family::rope::scaling::rope_scaling;

/// PLaMo 3's output norms' offsets (`modeling_plamo.py`,
/// `PLAMO3_POST_MIXER_NORM_OFFSET = 1.0 / 5` and
/// `PLAMO3_POST_MLP_NORM_OFFSET = 1.0 / (5**1.5)`).
pub(in crate::runtime::load::family) const PLAMO3_POST_MIXER_NORM_OFFSET: f64 = 0.2;

pub(in crate::runtime::load::family) const PLAMO3_POST_MLP_NORM_OFFSET: f64 =
    0.089_442_719_099_991_59;

/// The YaRN band `Plamo3Config.rope_parameters` fixes for the
/// full-attention layers (`"beta_fast": 32.0`, `"beta_slow": 1.0`).
pub(in crate::runtime::load::family) const PLAMO3_YARN_BETA_FAST: f64 = 32.0;

pub(in crate::runtime::load::family) const PLAMO3_YARN_BETA_SLOW: f64 = 1.0;

/// PLaMo 3 (`plamo3`): Llama's block under its names (`model.layers.layers`,
/// `mixer.qkv_proj`, `mlp.gate_up_proj`) with sandwich norms — the pre-norms
/// and the final norm storing their scales as offsets from one, the output
/// norms from 1/5 and 5^-1.5 — and per-head query and key norms before the
/// rotation. Every `sliding_window_pattern`-th layer attends fully, the
/// others through `window_size` and rotate at `rope_local_theta`; the full
/// layers rotate at `rope_theta`, by YaRN over `initial_context_length`
/// positions without rounding its band when `rope_scaling_factor` is not
/// one. Under `scale_embedding` the embedding is multiplied by
/// `sqrt(hidden_size)`.
pub(in crate::runtime::load::family) fn plamo3(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    if text(raw, "linear_type").is_some_and(|kind| kind != "normal") {
        bail!(
            "{} declares a linear_type other than normal; Ster reads PLaMo 3's unquantized projections",
            path.display()
        );
    }
    let pattern = whole(raw, "sliding_window_pattern")
        .filter(|pattern| *pattern > 0)
        .with_context(|| {
            format!(
                "{} declares a PLaMo 3 model without a sliding_window_pattern above zero",
                path.display()
            )
        })?;
    fits(layers, path)?;
    architecture.names = Names::PLAMO3;
    architecture.qkv_layout = QkvLayout::Stacked;
    architecture.fused_feed_forward = true;
    architecture.norm_offset = true;
    architecture.output_norms = true;
    architecture.output_norm_shifts =
        Some((PLAMO3_POST_MIXER_NORM_OFFSET, PLAMO3_POST_MLP_NORM_OFFSET));
    architecture.query_key_norm = QueryKeyNorm::PerHead;
    architecture.sliding_window =
        whole(raw, "window_size").or_else(|| whole(raw, "sliding_window"));
    architecture.sliding_layers = (0..layers)
        .filter(|layer| (layer + 1) % pattern != 0)
        .fold(0u128, |set, layer| set | (1u128 << layer));
    architecture.local_rope_theta = number(raw, "rope_local_theta").map(|theta| theta as f32);
    if flag(raw, "scale_embedding") {
        architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
    }
    let factor = number(raw, "rope_scaling_factor").unwrap_or(1.0);
    if factor != 1.0 {
        let original = whole(raw, "initial_context_length").with_context(|| {
            format!(
                "{} scales its rotation without initial_context_length",
                path.display()
            )
        })?;
        let yarn = serde_json::json!({
            "rope_type": "yarn",
            "factor": factor,
            "beta_fast": PLAMO3_YARN_BETA_FAST,
            "beta_slow": PLAMO3_YARN_BETA_SLOW,
            "original_max_position_embeddings": original,
            "truncate": false,
        });
        architecture.rope_scaling =
            rope_scaling(Some(&yarn), architecture.rotary_dim, raw, llama, path)?;
    }
    Ok(())
}

/// MuseGlimmer's text decoder (`muse_glimmer_text`, inside `muse_glimmer`,
/// whose weights sit below `model.language_model` and whose `lm_head` sits
/// at the root): Gemma 2's sandwich norms storing their scale as an offset
/// from one, the output norms at `post_norm_eps`, and a final norm storing
/// its scale as is; a weightless RMS norm after the embedding; a weightless
/// per-head norm over every query and key unless `use_qk_norm` is false,
/// the query then multiplied by `scale_query_by` (else `qk_scale_factor`,
/// divided by `sqrt(head_dim)` when it is at least that, as vLLM's
/// `_muse_glimmer_query_prescale` reads both schemas); attention's output
/// multiplied by the sigmoid of `self_attn.gate_proj` unless
/// `use_attn_output_gate` is false; sliding-window layers as `layer_types`
/// lists, the bases of `layer_rope_theta` with zero for the unrotated
/// full-attention layers; and the logits multiplied by `output_multiplier`
/// and capped at `final_logit_softcapping`.
pub(in crate::runtime::load::family) fn muse_glimmer(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(types) = raw.get("layer_types").and_then(Value::as_array) else {
        bail!(
            "{} declares a MuseGlimmer model without layer_types",
            path.display()
        );
    };
    architecture.names = Names::MUSE_GLIMMER;
    architecture.norm_offset = true;
    architecture.plain_final_norm = true;
    architecture.output_norms = true;
    architecture.output_norm_eps = number(raw, "post_norm_eps");
    architecture.embedding_norm = true;
    architecture.sliding_window = whole(raw, "sliding_window");
    architecture.sliding_layers = listed_layers(types, layers, path)?;
    layer_bases(raw, layers, llama, architecture, path)?;
    if raw.get("use_qk_norm").and_then(Value::as_bool) != Some(false) {
        architecture.query_key_norm = QueryKeyNorm::Weightless;
        let head_dim = architecture.head_dim as f64;
        let prescale = match (
            number(raw, "scale_query_by"),
            number(raw, "qk_scale_factor"),
        ) {
            (Some(explicit), _) => explicit,
            (None, Some(factor)) if factor >= head_dim.sqrt() => factor / head_dim.sqrt(),
            (None, Some(factor)) => factor,
            (None, None) => 1.0,
        };
        if prescale <= 0.0 {
            bail!(
                "{} scales its queries by {prescale}; the scale must be above zero",
                path.display()
            );
        }
        architecture.score_divisor = head_dim.sqrt() / prescale;
    }
    if raw.get("use_attn_output_gate").and_then(Value::as_bool) != Some(false) {
        architecture.attention_gate = Some(GateFunction::Sigmoid);
    }
    architecture.logits_multiplier = number(raw, "output_multiplier");
    architecture.final_softcap = number(raw, "final_logit_softcapping");
    Ok(())
}
