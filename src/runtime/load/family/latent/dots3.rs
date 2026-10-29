//! Dots3-Note.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::{Architecture, GateFunction, LatentAttention, RopeScaling, WindowLatent};
use crate::runtime::load::family::read::{flag, number, text, whole};

/// Dots3-Note (`dots3_note`, vLLM's `Dots3NoteForCausalLM`): DeepSeek-V3.2's
/// block whose `full_attention` layers keep the sparse-attention indexer and
/// whose `sliding_attention` layers run a dense latent attention of their
/// own through `sliding_window_size` ([`WindowLatent`]: the `swa_*` heads,
/// ranks and head parts, rotated by `swa_rope_theta`). Every latent
/// attention norms its shared rotated key part (`k_rope_only_layernorm`),
/// multiplies its bottlenecks' normed outputs by `sqrt(hidden_size / rank)`
/// under `apply_mla_qkv_lora_rescale`, and gates each head's output by the
/// sigmoid of its `g_proj` logit (`attention_gate_type` `headwise`). vLLM
/// rotates both kinds by their plain base, whatever `rope_scaling` states.
pub(in crate::runtime::load::family) fn dots3_note(
    raw: &Value,
    llama: &LlamaConfig,
    architecture: &mut Architecture,
    path: &Path,
) -> Result<()> {
    let part = |key: &str| -> Result<usize> {
        whole(raw, key).with_context(|| {
            format!(
                "{} declares a Dots3-Note model without {key}",
                path.display()
            )
        })
    };
    let Some(full) = architecture
        .latent
        .filter(|latent| latent.query_rank.is_some())
    else {
        bail!(
            "{} declares Dots3-Note attention without q_lora_rank; its latent attention always has a query bottleneck",
            path.display()
        );
    };
    let window = LatentAttention {
        query_rank: Some(part("swa_q_lora_rank")?),
        key_value_rank: part("swa_kv_lora_rank")?,
        unrotated: part("swa_qk_nope_head_dim")?,
        rotated: part("swa_qk_rope_head_dim")?,
        value: part("swa_v_head_dim")?,
    };
    if window.rotated != full.rotated {
        bail!(
            "{} rotates {} components per sliding-window head and {} per full-attention head; Ster builds both rotations at one width",
            path.display(),
            window.rotated,
            full.rotated
        );
    }
    for key in ["attention_gate_type", "swa_attention_gate_type"] {
        match text(raw, key) {
            Some("headwise") => {}
            other => bail!(
                "{} declares {key} {other:?}; Ster implements Dots3-Note's headwise gate",
                path.display()
            ),
        }
    }
    architecture.head_gate = Some(GateFunction::Sigmoid);
    architecture.rotated_key_norm = true;
    let hidden = llama.hidden_size as f64;
    let rescale = flag(raw, "apply_mla_qkv_lora_rescale");
    let scales = |latent: LatentAttention| {
        latent.query_rank.filter(|_| rescale).map(|rank| {
            (
                (hidden / rank as f64).sqrt(),
                (hidden / latent.key_value_rank as f64).sqrt(),
            )
        })
    };
    architecture.latent_scales = scales(full);
    architecture.window_latent = Some(WindowLatent {
        heads: part("swa_num_attention_heads")?,
        latent: window,
        scales: scales(window),
    });
    architecture.sliding_window =
        whole(raw, "sliding_window_size").or_else(|| whole(raw, "sliding_window"));
    architecture.local_rope_theta = number(raw, "swa_rope_theta")
        .filter(|base| *base != f64::from(llama.rope_theta))
        .map(|base| base as f32);
    architecture.rope_scaling = RopeScaling::None;
    architecture.score_divisor = (architecture.head_dim as f64).sqrt();
    Ok(())
}
