//! The rotary scaling [`super::take_rope_scaling`] took, as Ster applies it.

use std::path::Path;

use anyhow::Result;
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::RopeScaling;
use crate::runtime::load::family::read::{whole, within, within_or};
use crate::runtime::load::family::rope::scaling_kind;

/// Where the scaling sits in the config, as a refusal of one of its keys
/// names it.
const SECTION: &str = "rope_scaling";

/// The scaling [`take_rope_scaling`] took, over `rotary_dim` components.
pub(in crate::runtime::load::family) fn rope_scaling(
    scaling: Option<&Value>,
    rotary_dim: usize,
    raw: &Value,
    llama: &LlamaConfig,
    path: &Path,
) -> Result<RopeScaling> {
    let Some(scaling) = scaling else {
        return Ok(RopeScaling::None);
    };
    match scaling_kind(scaling, "rope_scaling", path)? {
        "linear" => {
            let Some(factor) = within::<f64>(scaling, "rope_scaling", "factor", path)? else {
                refused!(
                    Incomplete,
                    ["rope_scaling" => scaling],
                    "{} declares linear rope_scaling with no factor",
                    path.display()
                );
            };
            Ok(RopeScaling::Linear(factor as f32))
        }
        "longrope" => {
            let factors = |key: &str| -> Result<Vec<f32>> {
                let values: Option<Vec<f32>> = within::<&[Value]>(scaling, SECTION, key, path)?
                    .and_then(|list| list.iter().map(|v| v.as_f64().map(|v| v as f32)).collect());
                match values {
                    Some(values) if values.len() == rotary_dim / 2 => Ok(values),
                    _ => refused!(
                        Malformed,
                        ["rope_scaling" => scaling],
                        "{} declares longrope {key} that is not {} numbers, one per rotated frequency",
                        path.display(),
                        rotary_dim / 2
                    ),
                }
            };
            let (short, long) = (factors("short_factor")?, factors("long_factor")?);
            // Phi-3 states the original context at the top level and derives
            // the factor from it, as Transformers does.
            let maximum = llama.max_position_embeddings;
            let (original, factor) = match whole(raw, "original_max_position_embeddings", path)? {
                Some(original) => (original, maximum as f64 / original as f64),
                None => (maximum, within_or(scaling, SECTION, "factor", 1.0, path)?),
            };
            let attention = match within::<f64>(scaling, SECTION, "attention_factor", path)? {
                Some(attention) => attention,
                None if factor <= 1.0 => 1.0,
                None => (1.0 + factor.ln() / (original as f64).ln()).sqrt(),
            };
            // PhiMoE states each table's magnitude itself.
            Ok(RopeScaling::LongRope {
                short,
                long,
                original,
                short_attention: within_or(scaling, SECTION, "short_mscale", attention, path)?
                    as f32,
                long_attention: within_or(scaling, SECTION, "long_mscale", attention, path)?
                    as f32,
            })
        }
        // TeleChat3's `telechat3-yarn` is YaRN with a gentler temperature
        // slope (its `_compute_telechat_yarn_parameters`); vLLM's
        // `deepseek_yarn` (Sarvam) is YaRN with DeepSeek's `mscale` ratio.
        kind @ ("yarn" | "telechat3-yarn" | "deepseek_yarn") => {
            let slope = if kind == "telechat3-yarn" {
                TELECHAT3_YARN_SLOPE
            } else {
                YARN_SLOPE
            };
            let Some(factor) = within::<f64>(scaling, "rope_scaling", "factor", path)? else {
                refused!(
                    Incomplete,
                    ["rope_scaling" => scaling],
                    "{} declares yarn rope_scaling with no factor",
                    path.display()
                );
            };
            let original = within_or(
                scaling,
                SECTION,
                "original_max_position_embeddings",
                llama.max_position_embeddings,
                path,
            )?;
            // Transformers reads a zero `mscale`, `mscale_all_dim`,
            // `attention_factor` or beta as not stated (`if mscale and
            // mscale_all_dim`).
            let stated = |key: &str| -> Result<Option<f64>> {
                Ok(within::<f64>(scaling, SECTION, key, path)?.filter(|v| *v != 0.0))
            };
            // Transformers' `_compute_yarn_parameters`: the cos and sin
            // magnitude is `mscale / mscale_all_dim` when both are stated,
            // otherwise the stated `attention_factor`, otherwise YaRN's own.
            let attention = match (stated("mscale")?, stated("mscale_all_dim")?) {
                (Some(mscale), Some(all)) => {
                    yarn_mscale(factor, mscale, slope) / yarn_mscale(factor, all, slope)
                }
                _ => match stated("attention_factor")? {
                    Some(attention) => attention,
                    None => yarn_mscale(factor, 1.0, slope),
                },
            };
            let beta = |key: &str, default: f64| -> Result<f32> {
                let beta = match stated(key)? {
                    Some(beta) => beta,
                    None => default,
                };
                Ok(beta as f32)
            };
            Ok(RopeScaling::Yarn {
                factor: factor as f32,
                original,
                beta_fast: beta("beta_fast", YARN_BETA_FAST)?,
                beta_slow: beta("beta_slow", YARN_BETA_SLOW)?,
                attention: attention as f32,
                truncate: within_or(scaling, "rope_scaling", "truncate", true, path)?,
            })
        }
        "proportional" => proportional(scaling, rotary_dim, path),
        _ => Ok(RopeScaling::None),
    }
}

/// Gemma 4's proportional rotation over `width` components, as
/// Transformers' `_compute_proportional_rope_parameters` computes it: the
/// first `partial_rotary_factor · width / 2` frequency pairs, rounded down,
/// rotate at `rope_theta^(-2i / width)`, the rest are zero, and every
/// frequency is divided by `factor`.
pub(in crate::runtime::load::family) fn proportional(
    scaling: &Value,
    width: usize,
    path: &Path,
) -> Result<RopeScaling> {
    let share = within_or(scaling, "rope_scaling", "partial_rotary_factor", 1.0, path)?;
    Ok(RopeScaling::Proportional {
        rotated: (share * width as f64 / 2.0).floor() as usize,
        factor: within_or(scaling, "rope_scaling", "factor", 1.0, path)? as f32,
    })
}

/// YaRN's default rotation counts bounding the interpolated band (Peng et
/// al., 2023; Transformers' `beta_fast` and `beta_slow`).
pub(in crate::runtime::load::family) const YARN_BETA_FAST: f64 = 32.0;

pub(in crate::runtime::load::family) const YARN_BETA_SLOW: f64 = 1.0;

/// YaRN's temperature slope (Peng et al., 2023; Transformers'
/// `get_mscale`), and TeleChat3's (`get_mscale` in its
/// `modeling_telechat3.py`).
pub(in crate::runtime::load::family) const YARN_SLOPE: f64 = 0.1;

pub(in crate::runtime::load::family) const TELECHAT3_YARN_SLOPE: f64 = 0.07;

/// YaRN's attention temperature for a context stretched by `factor`:
/// `slope * mscale * ln(factor) + 1`, or one when nothing is stretched.
pub(in crate::runtime::load::family) fn yarn_mscale(factor: f64, mscale: f64, slope: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        slope * mscale * factor.ln() + 1.0
    }
}
