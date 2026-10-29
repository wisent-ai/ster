//! Rotary bases stated per layer kind.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::Architecture;

/// `layer_rope_theta`, a base per layer where zero means no rotation
/// (GraniteSWA, MuseGlimmer). Ster rotates full-attention layers by
/// `rope_theta` and sliding-window layers by one base of their own, so the
/// stated bases must fit that.
pub(in crate::runtime::load::family) fn layer_bases(
    raw: &Value,
    layers: usize,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let Some(thetas) = raw.get("layer_rope_theta").filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let thetas: Vec<f64> = thetas
        .as_array()
        .and_then(|list| list.iter().map(Value::as_f64).collect())
        .with_context(|| {
            format!(
                "{} declares layer_rope_theta that is not a list of numbers",
                path.display()
            )
        })?;
    if thetas.len() != layers {
        bail!(
            "{} lists {} layer_rope_theta values for {layers} layers",
            path.display(),
            thetas.len()
        );
    }
    let mut local: Option<f64> = None;
    for (layer, theta) in thetas.into_iter().enumerate() {
        let sliding = architecture.window(layer).is_some();
        if theta == 0.0 {
            architecture.unrotated_layers |= 1u128 << layer;
        } else if !sliding && theta != f64::from(llama.rope_theta) {
            bail!(
                "{} rotates full-attention layer {layer} by base {theta}, not rope_theta {}; Ster rotates every full-attention layer by rope_theta",
                path.display(),
                llama.rope_theta
            );
        } else if sliding && local.is_some_and(|base| base != theta) {
            bail!(
                "{} rotates sliding-window layers by more than one base in layer_rope_theta; Ster rotates them all by one",
                path.display()
            );
        } else if sliding {
            local = Some(theta);
        }
    }
    architecture.local_rope_theta = local
        .filter(|base| *base != f64::from(llama.rope_theta))
        .map(|base| base as f32);
    Ok(())
}
