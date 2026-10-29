//! The shape of each attention head and how much of it rotates.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, LatentAttention, Positions, RopeScaling};
use crate::runtime::load::family::read::{number, whole};
use crate::runtime::load::family::rope::scaling::{YARN_SLOPE, rope_scaling, yarn_mscale};

/// The head's width, its latent attention when the config states one, how much
/// of each head rotates, and the scaling of that rotation.
pub(in crate::runtime::load::family) fn head(
    raw: &Value,
    scaling: Option<&Value>,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    if let Some(head_dim) = whole(raw, "head_dim") {
        architecture.head_dim = head_dim;
        architecture.score_divisor = (head_dim as f64).sqrt();
    }
    // Multi-head latent attention (DeepSeek-V2 and V3, MiniCPM3) states its
    // bottlenecks and head parts; a head is its position-free part plus its
    // rotated part, and only the rotated part rotates.
    if let Some(key_value_rank) = whole(raw, "kv_lora_rank") {
        let part = |key: &str| -> Result<usize> {
            whole(raw, key)
                .with_context(|| format!("{} declares kv_lora_rank without {key}", path.display()))
        };
        let latent = LatentAttention {
            query_rank: whole(raw, "q_lora_rank"),
            key_value_rank,
            unrotated: part("qk_nope_head_dim")?,
            rotated: part("qk_rope_head_dim")?,
            value: part("v_head_dim")?,
        };
        architecture.head_dim = latent.unrotated + latent.rotated;
        architecture.score_divisor = (architecture.head_dim as f64).sqrt();
        architecture.latent = Some(latent);
    }
    // The rotated share of each head: `partial_rotary_factor` (or GPT-NeoX's
    // `rotary_pct`) as a fraction, which Transformers truncates, GPT-J's
    // `rotary_dim` as a width, or latent attention's rotated part.
    let head_dim = architecture.head_dim;
    let fraction = number(raw, "partial_rotary_factor").or_else(|| number(raw, "rotary_pct"));
    architecture.rotary_dim = match (architecture.latent, fraction, whole(raw, "rotary_dim")) {
        (Some(latent), _, _) => latent.rotated,
        (None, Some(factor), _) => (head_dim as f64 * factor) as usize,
        (None, None, Some(width)) => width,
        (None, None, None) => head_dim,
    };
    let rotary_dim = architecture.rotary_dim;
    // A latent attention with no rotated part (GLM-5-Next's
    // `qk_rope_head_dim` 0) rotates nothing: its positions come from the
    // causal order alone.
    let unrotated_latent = architecture
        .latent
        .is_some_and(|latent| latent.rotated == 0);
    if unrotated_latent {
        architecture.positions = Positions::None;
        architecture.rotary_dim = head_dim;
    } else if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
        bail!(
            "{} rotates {rotary_dim} of {head_dim} components per head (partial_rotary_factor, rotary_pct or rotary_dim); the rotated width must be even, above zero and at most the head",
            path.display()
        );
    }
    architecture.rope_scaling = rope_scaling(scaling, architecture.rotary_dim, raw, llama, path)?;
    // DeepSeek sharpens the scores of a YaRN-stretched model by the square
    // of the temperature `mscale_all_dim` gives.
    if architecture.latent.is_some() {
        let all = scaling
            .and_then(|scaling| scaling.get("mscale_all_dim"))
            .and_then(Value::as_f64)
            .filter(|value| *value != 0.0);
        if let (RopeScaling::Yarn { factor, .. }, Some(all)) = (&architecture.rope_scaling, all) {
            let temperature = yarn_mscale(f64::from(*factor), all, YARN_SLOPE);
            architecture.score_divisor /= temperature * temperature;
        }
    }
    Ok(())
}
