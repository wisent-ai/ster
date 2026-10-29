//! DeepSeek Sparse Attention's indexer.

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::model::{IndexerSpec, KeyPool};
use crate::runtime::load::family::read::layers::fits;
use crate::runtime::load::family::read::{flag, whole};

/// DeepSeek Sparse Attention's defaults when a config leaves them out
/// (Transformers' `DeepseekV32Config`: `index_topk` 2048, `index_head_dim`
/// 128, `index_n_heads` 64; `GlmMoeDsaConfig` narrows the heads to 32;
/// `index_topk_freq` 1 and `index_skip_topk_offset` 2).
pub(in crate::runtime::load::family) const INDEX_TOP_K: usize = 2048;

pub(in crate::runtime::load::family) const INDEX_HEAD_DIM: usize = 128;

pub(in crate::runtime::load::family) const DEEPSEEK_INDEX_HEADS: usize = 64;

pub(in crate::runtime::load::family) const GLM_INDEX_HEADS: usize = 32;

pub(in crate::runtime::load::family) const INDEX_FREQUENCY: usize = 1;

pub(in crate::runtime::load::family) const INDEX_SKIP_OFFSET: usize = 2;

/// HY-V4's indexer heads when its config leaves `index_n_heads` out
/// (`HYV4Config`, https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_INDEX_HEADS: usize = 16;

/// How often HY-V4's layers index when its config lists no `indexer_types`:
/// layer 0 and every fourth layer from layer 1, which is GLM-5's rule with
/// this frequency and its offset (`HYV4Config.__post_init__`,
/// https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/hy_v4.py).
pub(in crate::runtime::load::family) const HY_V4_INDEX_EVERY: usize = 4;

/// DeepSeek Sparse Attention's indexer from the config. GLM-5 and HY-V4 mark
/// each layer `full` or `shared` in `indexer_types` (GLM-5 also as `F` and
/// `S` in `index_topk_pattern`), or derive it from `index_topk_freq` and
/// `index_skip_topk_offset` as `GlmMoeDsaConfig.__post_init__` does: layer
/// `i` indexes when `max(i − offset + 1, 0)` is a multiple of the frequency.
/// HY-V4's indexer rotates adjacent pairs at the end of each head.
pub(in crate::runtime::load::family) fn sparse_index(
    raw: &Value,
    model_type: &str,
    layers: usize,
    path: &Path,
) -> Result<IndexerSpec> {
    fits(layers, path)?;
    let glm = matches!(model_type, "glm_moe_dsa" | "glm5_next_text");
    let hy = model_type == "hy_v4";
    let mut shared_layers = 0u128;
    if glm || hy {
        let listed: Option<Vec<bool>> = match (raw.get("indexer_types"), raw.get("index_topk_pattern")) {
            (Some(Value::Array(kinds)), _) => Some(
                kinds
                    .iter()
                    .enumerate()
                    .map(|(layer, kind)| match kind.as_str() {
                        Some("full") => Ok(false),
                        Some("shared") => Ok(true),
                        other => bail!(
                            "{} names layer {layer}'s indexer {other:?}; an indexer is full or shared",
                            path.display()
                        ),
                    })
                    .collect::<Result<_>>()?,
            ),
            (_, Some(Value::String(pattern))) => Some(
                pattern
                    .chars()
                    .enumerate()
                    .map(|(layer, mark)| match mark {
                        'F' => Ok(false),
                        'S' => Ok(true),
                        other => bail!(
                            "{} marks layer {layer}'s indexer {other:?} in index_topk_pattern; a mark is F or S",
                            path.display()
                        ),
                    })
                    .collect::<Result<_>>()?,
            ),
            _ => None,
        };
        let shared: Vec<bool> = match listed {
            Some(shared) => shared,
            None => {
                let frequency = match whole(raw, "index_topk_freq") {
                    Some(stated) => stated,
                    None if hy => HY_V4_INDEX_EVERY,
                    None => INDEX_FREQUENCY,
                }
                .max(1);
                let offset = whole(raw, "index_skip_topk_offset").unwrap_or(INDEX_SKIP_OFFSET);
                (0..layers)
                    .map(|layer| (layer + 1).saturating_sub(offset) % frequency != 0)
                    .collect()
            }
        };
        if shared.len() != layers {
            bail!(
                "{} marks {} layers' indexers for {layers} layers",
                path.display(),
                shared.len()
            );
        }
        if shared.first() == Some(&true) {
            bail!(
                "{} makes layer 0's indexer shared, with no earlier layer to share from",
                path.display()
            );
        }
        shared_layers = shared
            .iter()
            .enumerate()
            .filter(|(_, shared)| **shared)
            .fold(0u128, |set, (layer, _)| set | (1u128 << layer));
    }
    let pool = if model_type == "glm5_next_text" {
        let Some(size) = whole(raw, "index_kpool").filter(|size| *size > 0) else {
            bail!(
                "{} declares a GLM-5-Next indexer without index_kpool",
                path.display()
            );
        };
        Some(KeyPool {
            size,
            tail: flag(raw, "index_kpool_always_select_tail"),
        })
    } else {
        None
    };
    Ok(IndexerSpec {
        heads: match whole(raw, "index_n_heads") {
            Some(stated) => stated,
            None if hy => HY_V4_INDEX_HEADS,
            None if glm => GLM_INDEX_HEADS,
            None => DEEPSEEK_INDEX_HEADS,
        },
        head_dim: whole(raw, "index_head_dim").unwrap_or(INDEX_HEAD_DIM),
        top_k: whole(raw, "index_topk").unwrap_or(INDEX_TOP_K),
        // vLLM's HY-V4 indexer rotates adjacent pairs (`is_neox_style=False`):
        // its checkpoints are in Megatron's layout.
        interleaved: hy
            || (glm || model_type == "dots3_note")
                && raw
                    .get("indexer_rope_interleave")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
        shared_layers,
        pool,
        rotate_tail: hy,
    })
}
