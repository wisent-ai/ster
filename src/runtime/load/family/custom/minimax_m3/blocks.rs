//! MiniMax Sparse Attention's block indexer, read from
//! `sparse_attention_config`.

use std::num::NonZeroUsize;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::model::BlockIndexSpec;
use crate::runtime::load::family::read::layers::flagged_layers;
use crate::runtime::load::family::read::{flag_or, stated, text, whole};

/// The block indexer's heads each lead a group of query heads and rotate
/// by the attention's angles: a query head count the index heads do not
/// divide, or a rotation wider than an index head, is refused here rather
/// than at the first forward pass that ranks key blocks.
pub(super) fn index_fits(
    raw: &Value,
    index: BlockIndexSpec,
    query_heads: usize,
    rotary_dim: usize,
    path: &Path,
) -> Result<()> {
    if index.heads == 0 || query_heads % index.heads != 0 {
        refused!(
            Contradictory,
            in raw,
            ["num_attention_heads", "sparse_attention_config.sparse_num_index_heads"],
            "{} declares {query_heads} query heads and {} index heads; MiniMax-M3's index heads each choose the blocks of an equal group of query heads",
            path.display(),
            index.heads
        );
    }
    if rotary_dim > index.head_dim {
        refused!(
            Contradictory,
            in raw,
            [
                "partial_rotary_factor",
                "head_dim",
                "sparse_attention_config.sparse_index_dim"
            ],
            "{} rotates {rotary_dim} components of each head and its index heads are {} wide; the block indexer rotates its heads by the attention's angles",
            path.display(),
            index.head_dim
        );
    }
    Ok(())
}

/// MiniMax Sparse Attention's block indexer from `sparse_attention_config`,
/// or `None` when the config runs every layer densely.
pub(super) fn block_index(
    raw: &Value,
    layers: usize,
    path: &Path,
) -> Result<Option<BlockIndexSpec>> {
    const SPARSE: &str = "sparse_attention_config";
    let Some(sparse) = stated::<&serde_json::Map<String, Value>>(raw, SPARSE, path)? else {
        return Ok(None);
    };
    if !flag_or(raw, "sparse_attention_config.use_sparse_attention", true, path)? {
        return Ok(None);
    }
    let indexed = flagged_layers(
        raw,
        "sparse_attention_config.sparse_attention_freq",
        layers,
        path,
    )?;
    match text(raw, "sparse_attention_config.sparse_score_type", path)? {
        None | Some("max") => {}
        Some(other) => refused!(
            Unimplemented,
            in raw,
            ["sparse_attention_config.sparse_score_type"],
            "{} declares sparse_score_type {other:?}; Ster scores a key block by its best key (max)",
            path.display()
        ),
    }
    if whole(raw, "sparse_attention_config.sparse_init_block", path)?
        .and_then(NonZeroUsize::new)
        .is_some()
    {
        refused!(
            Unimplemented,
            in raw,
            ["sparse_attention_config.sparse_init_block"],
            "{} keeps sparse_init_block leading blocks always visible; Ster implements the local blocks Transformers' MiniMax-M3 indexer keeps, not leading ones",
            path.display()
        );
    }
    // Every sparse layer of MiniMax-M3 disables the indexer's own value and
    // output branch; neither vLLM nor Transformers implements it.
    if sparse.contains_key("sparse_disable_index_value") {
        let disabled = flagged_layers(
            raw,
            "sparse_attention_config.sparse_disable_index_value",
            layers,
            path,
        )?;
        if indexed & disabled != indexed {
            refused!(
                Unimplemented,
                in raw,
                [
                    "sparse_attention_config.sparse_attention_freq",
                    "sparse_attention_config.sparse_disable_index_value",
                ],
                "{} enables the indexer's value branch on a sparse layer (sparse_disable_index_value); Ster implements MiniMax-M3's indexer without it",
                path.display()
            );
        }
    }
    let size = |key: &str| -> Result<usize> {
        match whole(raw, &format!("{SPARSE}.{key}"), path)? {
            Some(size) => Ok(size),
            None => refused!(
                Incomplete,
                in raw,
                [format!("sparse_attention_config.{key}")],
                "{} declares sparse_attention_config without {key}",
                path.display()
            ),
        }
    };
    let heads = size("sparse_num_index_heads")?;
    if Some(heads) != whole(raw, "num_key_value_heads", path)? {
        refused!(
            Contradictory,
            in raw,
            ["sparse_attention_config.sparse_num_index_heads", "num_key_value_heads"],
            "{} declares {heads} index heads (sparse_num_index_heads) and a different num_key_value_heads; MiniMax-M3 chooses one block selection per key-value head",
            path.display()
        );
    }
    let (Some(block), Some(top_blocks)) = (
        NonZeroUsize::new(size("sparse_block_size")?),
        NonZeroUsize::new(size("sparse_topk_blocks")?),
    ) else {
        refused!(
            OutOfRange,
            in raw,
            [
                "sparse_attention_config.sparse_block_size",
                "sparse_attention_config.sparse_topk_blocks",
            ],
            "{} declares an empty sparse_block_size or sparse_topk_blocks",
            path.display()
        );
    };
    let local_blocks = size("sparse_local_block")?;
    if local_blocks > top_blocks.get() {
        refused!(
            Contradictory,
            in raw,
            [
                "sparse_attention_config.sparse_local_block",
                "sparse_attention_config.sparse_topk_blocks",
            ],
            "{} keeps {local_blocks} local key blocks (sparse_local_block) among only {top_blocks} kept ones (sparse_topk_blocks)",
            path.display()
        );
    }
    Ok(Some(BlockIndexSpec {
        heads,
        head_dim: size("sparse_index_dim")?,
        block,
        top_blocks: top_blocks.get(),
        local_blocks,
        layers: indexed,
    }))
}

