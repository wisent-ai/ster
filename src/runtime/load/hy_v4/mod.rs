//! HY-V4's release tensor names, answering each Transformers name Ster asks
//! for. Transformers reads an HY-V4 checkpoint through the renames of its
//! `hy_v4` conversion mapping (`conversion_mapping.py`); this applies the
//! same renames to every name the checkpoint stores and keeps the result
//! the other way round, so the release's layout (`hc_pre` inside each
//! hyper-connection module) and the one Transformers' own writer produces
//! (`hc_pre` outside it, `linear_gate` on the feed-forwards too) both read
//! alike.

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use candle_core::safetensors::MmapedSafetensors;

/// Transformers' `hy_v4` renames in the order it applies them: each
/// fragment of a stored name and what it becomes.
const RENAMES: &[(&str, &str)] = &[
    // The prefix every hyper-connection parameter carries is dropped.
    (".hc_pre.hc_", ".hc_"),
    // The attention's sinks and output gate, and the feed-forward gates.
    (".learnable_sink_param", ".sinks"),
    (".linear_gate", ".gate_proj"),
    // Each layer's hyper-connections, as DeepSeek-V4 names them.
    (".hc_attn_layer.hc_fn", ".attn_hc.fn"),
    (".hc_attn_layer.hc_base", ".attn_hc.base"),
    (".hc_attn_layer.hc_scale", ".attn_hc.scale"),
    (".hc_mlp_layer.hc_fn", ".ffn_hc.fn"),
    (".hc_mlp_layer.hc_base", ".ffn_hc.base"),
    (".hc_mlp_layer.hc_scale", ".ffn_hc.scale"),
    // The model's last collapse of its streams.
    (".hc_head_fn", ".hc_fn"),
    (".hc_head_base", ".hc_base"),
    (".hc_head_scale", ".hc_scale"),
];

/// Every Transformers name the checkpoint's tensors answer to, with the name
/// each is stored under.
pub(super) struct StoredNames {
    stored: HashMap<String, String>,
}

impl StoredNames {
    /// Reads the names `weights` store, without reading a tensor.
    pub(super) fn read(weights: &[PathBuf]) -> Result<Self> {
        let files = unsafe { MmapedSafetensors::multi(weights) }.with_context(|| {
            format!(
                "failed to read the tensor names of {} HY-V4 weight files",
                weights.len()
            )
        })?;
        let stored = files
            .tensors()
            .into_iter()
            .map(|(name, _)| (transformers_name(&name), name))
            .collect();
        Ok(Self { stored })
    }

    /// The stored name of the Transformers name `name`. A name no stored
    /// tensor renames to is asked for as it is, so a tensor the checkpoint
    /// lacks is refused under the name Ster wanted.
    pub(super) fn stored(&self, name: &str) -> String {
        match self.stored.get(name) {
            Some(stored) => stored.clone(),
            None => name.to_owned(),
        }
    }
}

/// The name Transformers gives the stored tensor `stored`.
fn transformers_name(stored: &str) -> String {
    RENAMES
        .iter()
        .fold(stored.to_owned(), |name, (from, to)| name.replace(from, to))
}
