//! The mixers that carry order in a recurrent state rather than a key-value
//! cache: Mamba's selective scan and LFM2's short convolution
//! (`state_space`), Mamba-2's structured scan (`structured`), and Qwen3-Next's
//! gated delta rule (`delta`). Each keeps its decode state in the cache's
//! recurrent-state slot for its layer.

pub(super) mod delta;
pub(super) mod state_space;
pub(super) mod structured;
