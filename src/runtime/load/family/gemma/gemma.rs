//! Gemma 1, 2 and 3, VaultGemma and Rnj-1.5.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{Activation, Architecture, Names, QueryKeyNorm};
use crate::runtime::load::family::At;
use crate::runtime::load::family::read::layers::{even_layers, fits};
use crate::runtime::load::family::read::{number, whole};

/// What the families this file reads add to the Llama block; [`super::super::family`]
/// sends only their `model_type` values here.
pub(in crate::runtime::load::family) fn configure(
    at: &At,
    architecture: &mut Architecture,
) -> Result<()> {
    let At {
        model_type,
        raw,
        llama,
        layers,
        path,
        ..
    } = *at;
    match model_type {
        // VaultGemma is Gemma 2 without the norms over each sublayer's
        // output.
        "gemma" | "gemma2" | "gemma3_text" | "vaultgemma" => {
            architecture.norm_offset = true;
            architecture.embedding_multiplier = Some((llama.hidden_size as f64).sqrt());
            architecture.activation = Activation::GeluTanh;
            if model_type != "gemma" {
                architecture.output_norms = model_type != "vaultgemma";
                architecture.names = Names::GEMMA2;
                architecture.sliding_window = whole(raw, "sliding_window");
                if let Some(scalar) = number(raw, "query_pre_attn_scalar") {
                    architecture.score_divisor = scalar.sqrt();
                }
                architecture.attention_softcap = number(raw, "attn_logit_softcapping");
                architecture.final_softcap = number(raw, "final_logit_softcapping");
            }
            if model_type == "gemma2" || model_type == "vaultgemma" {
                architecture.sliding_layers = even_layers(layers, path)?;
            }
            if model_type == "gemma3_text" {
                architecture.query_key_norm = QueryKeyNorm::PerHead;
                architecture.local_rope_theta =
                    number(raw, "rope_local_base_freq").map(|theta| theta as f32);
                // Rnj-1.5 lists `chunked_attention` layers: chunks of
                // `sliding_window` positions, each query seeing its own
                // chunk and the one before it.
                let chunked = raw
                    .get("layer_types")
                    .and_then(Value::as_array)
                    .is_some_and(|kinds| {
                        kinds
                            .iter()
                            .any(|kind| kind.as_str() == Some("chunked_attention"))
                    });
                if chunked {
                    architecture.chunk_lookback = Some(RNJ1_CHUNK_LOOKBACK);
                }
                // Older Gemma 3 configs state the pattern instead of
                // `layer_types`: every `sliding_window_pattern`-th layer is
                // global, the rest local.
                if let Some(pattern) = whole(raw, "sliding_window_pattern").filter(|p| *p > 0) {
                    fits(layers, path)?;
                    architecture.sliding_layers = (0..layers)
                        .filter(|layer| (layer + 1) % pattern != 0)
                        .fold(0, |set, layer| set | (1u128 << layer));
                }
            }
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    Ok(())
}

/// How many earlier chunks Rnj-1.5's chunked layers see (vLLM's `rnj1.py`,
/// `chunk_lookback = 1 if self.is_chunked`).
pub(in crate::runtime::load::family) const RNJ1_CHUNK_LOOKBACK: usize = 1;
