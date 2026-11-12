//! The sparse-attention indexers: DeepSeek Sparse Attention's and MiniMax's.

/// DeepSeek Sparse Attention's indexer: `heads` heads of `head_dim`
/// (`index_n_heads`, `index_head_dim`), keeping each query's `top_k`
/// (`index_topk`) keys, rotating adjacent pairs under `interleaved` (GLM-5's
/// `indexer_rope_interleave`) and halves otherwise; the layers in
/// `shared_layers` (GLM-5's `indexer_types` `shared`) have no indexer and
/// reuse the last indexed layer's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub interleaved: bool,
    pub shared_layers: u128,
    /// GLM-5-Next's key pools (`index_kpool`): the indexer scores pools of
    /// this many keys rather than keys.
    pub pool: Option<KeyPool>,
    /// The rotated components are each head's last `qk_rope_head_dim`
    /// rather than its first (HY-V4, whose checkpoints keep the indexer's
    /// position-free part first: vLLM's `hy_v4` `Indexer.prepare_inputs`).
    pub rotate_tail: bool,
}

/// GLM-5-Next's pooled indexer: every `size` consecutive keys are one
/// candidate, its key the softmax-weighted sum of theirs by
/// `index_kpool_compress_gate(x) + index_kpool_compress_ape`; a query keeps
/// its `top_k / size` best pools that ended at or before it, and under
/// `tail` (`index_kpool_always_select_tail`) the keys of its own unfinished
/// pool besides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPool {
    pub size: usize,
    pub tail: bool,
}

/// MiniMax Sparse Attention's block indexer (MiniMax-M3): on the layers in
/// `layers` (`sparse_attention_freq`), `heads` index heads of `head_dim`
/// (`sparse_num_index_heads`, one per key-value head, `sparse_index_dim`)
/// score every key against one shared index key, without a scale. A block
/// of `block` keys (`sparse_block_size`) scores as its best visible key, and
/// each query of a key-value group keeps its group's best `top_blocks`
/// blocks (`sparse_topk_blocks`), its own block and the `local_blocks - 1`
/// before it (`sparse_local_block`) always among them. Blocks are counted
/// from the first key slot, as Transformers' `MiniMaxM3VLIndexer` counts
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIndexSpec {
    pub heads: usize,
    pub head_dim: usize,
    pub block: std::num::NonZeroUsize,
    pub top_blocks: usize,
    pub local_blocks: usize,
    pub layers: u128,
}

impl BlockIndexSpec {
    pub fn indexes(&self, layer: usize) -> bool {
        layer < u128::BITS as usize && self.layers & (1u128 << layer) != 0
    }
}

impl IndexerSpec {
    pub fn shared(&self, layer: usize) -> bool {
        layer < u128::BITS as usize && self.shared_layers & (1u128 << layer) != 0
    }
}
