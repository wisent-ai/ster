//! Low-rank adapters: the only weights Ster ever trains, and the document that carries them.
//!
//! Ster's base model is mapped read-only from safetensors and never enters a
//! [`VarMap`], so the tensors registered here are by construction the complete
//! trainable set. An adapter is a pair `(a, b)` with `a: [rank, in]` and
//! `b: [out, rank]`; the update applied to a projection is
//! `scale * x @ a^T @ b^T`, which is the low-rank factorisation written in the
//! order that never materialises the dense `out x in` product.
//!
//! The live weights are in `adapter`; the document they are written to and
//! read back from is in `artifact`. What stays here is the shape an operator
//! asks for: which projections carry adapters, and at what rank.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

mod adapter;
mod artifact;

pub use adapter::{Adapter, Adapters};
pub use artifact::{Artifact, Kind, ARTIFACT_SCHEMA_VERSION, REWARD_HEAD_TENSOR};

/// The widths a model's projections have, which an adapter's factors must
/// match: the residual stream, the attention (query and output projection),
/// one key or value projection, and the feed-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widths {
    pub hidden: usize,
    pub attention: usize,
    pub key_value: usize,
    pub intermediate: usize,
}

/// Which projections carry adapters.
///
/// `Ord` is derived because [`Adapters`] keys a [`BTreeMap`] by
/// `(layer, target)`, which is what makes adapter construction — and therefore
/// the sequence of draws from the random initialiser — reproducible for a seed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Query,
    Key,
    Value,
    Output,
    Gate,
    Up,
    Down,
}

impl Target {
    /// Every target, in the order the refusal sentences and the CLI list them.
    pub const ALL: [Target; 7] = [
        Target::Query,
        Target::Key,
        Target::Value,
        Target::Output,
        Target::Gate,
        Target::Up,
        Target::Down,
    ];

    /// Reads a target from operator input.
    ///
    /// Both spellings are accepted: the short name Ster prints, and the
    /// Hugging Face projection name operators read off a model card. Neither is
    /// ambiguous, and refusing `q_proj` would only teach a translation table.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "query" | "q_proj" | "q" => Ok(Self::Query),
            "key" | "k_proj" | "k" => Ok(Self::Key),
            "value" | "v_proj" | "v" => Ok(Self::Value),
            "output" | "o_proj" | "o" => Ok(Self::Output),
            "gate" | "gate_proj" => Ok(Self::Gate),
            "up" | "up_proj" => Ok(Self::Up),
            "down" | "down_proj" => Ok(Self::Down),
            _ => bail!(
                "unknown adapter target {text:?}; expected one of query, key, value, output, gate, up, down"
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::Output => "output",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    /// The projection's weight name inside a Hugging Face Llama checkpoint.
    ///
    /// This is the other half of the naming contract: [`Adapter::tensor_names`]
    /// says where a factor lives in a Ster artifact, and this says which base
    /// tensor that factor is an update to. Merging is the one operation that
    /// needs both, and hard-coding the mapping at its call site would put half
    /// the contract in a file that does not own it.
    pub fn checkpoint_tensor(self, layer: usize) -> String {
        let leaf = match self {
            Self::Query => "self_attn.q_proj",
            Self::Key => "self_attn.k_proj",
            Self::Value => "self_attn.v_proj",
            Self::Output => "self_attn.o_proj",
            Self::Gate => "mlp.gate_proj",
            Self::Up => "mlp.up_proj",
            Self::Down => "mlp.down_proj",
        };
        format!("model.layers.{layer}.{leaf}.weight")
    }

    /// The `(outputs, inputs)` shape of the projection this target adapts.
    ///
    /// Grouped-query attention makes key and value narrower than query, a
    /// stated `head_dim` can make the attention width differ from the
    /// residual width, and the feed-forward block is wider than the residual
    /// stream, so the adapter factors are not square and cannot be derived
    /// from `hidden` alone.
    fn widths(self, widths: Widths) -> (usize, usize) {
        match self {
            Self::Query => (widths.attention, widths.hidden),
            Self::Output => (widths.hidden, widths.attention),
            Self::Key | Self::Value => (widths.key_value, widths.hidden),
            Self::Gate | Self::Up => (widths.intermediate, widths.hidden),
            Self::Down => (widths.hidden, widths.intermediate),
        }
    }

    /// Whether the projection reads the residual stream, so `a` is `[rank, hidden_size]`.
    /// The output projection reads the attention width instead, which a
    /// stated `head_dim` can make different.
    fn reads_hidden(self) -> bool {
        !matches!(self, Self::Down | Self::Output)
    }

    /// Whether the projection writes the residual stream, so `b` is `[hidden_size, rank]`.
    fn writes_hidden(self) -> bool {
        matches!(self, Self::Query | Self::Output | Self::Down)
    }
}

#[derive(Debug, Clone)]
pub struct Spec {
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<Target>,
    pub layers: Vec<usize>,
    /// Seeds the draw that fills `A`. Two runs of the same command must produce
    /// the same adapter, and Candle's initialiser draws from the device's own
    /// generator, so the seed has to reach the device before the first tensor
    /// is created rather than only shuffling the example order later.
    pub seed: u64,
}

impl Spec {
    /// Checks everything a spec can get wrong before a model is touched.
    ///
    /// An empty `layers` list is deliberately valid and means "every layer":
    /// the CLI parses `--layers all` before it knows how many layers the model
    /// has, so it hands the emptiness down and [`Spec::resolved`] expands it
    /// once the count is known. An empty `targets` list has no such excuse.
    pub fn validate(&self, layer_count: usize) -> Result<()> {
        if self.rank == 0 {
            bail!("adapter rank 0 trains nothing; choose a rank of at least 1");
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            bail!("adapter alpha {} is not a positive finite number", self.alpha);
        }
        if self.targets.is_empty() {
            bail!(
                "adapter spec names no targets; choose at least one of query, key, value, output, gate, up, down"
            );
        }
        for (index, target) in self.targets.iter().enumerate() {
            if self.targets[..index].contains(target) {
                bail!("adapter spec names target {} twice", target.name());
            }
        }
        for layer in &self.layers {
            if *layer >= layer_count {
                bail!(
                    "adapter spec names layer {layer}, but the model has {layer_count} layers"
                );
            }
        }
        Ok(())
    }

    /// The spec with `layers` pinned to concrete indices.
    ///
    /// Callers that hold the layer count run this once and pass the result
    /// everywhere, so the adapters that get built and the layer list recorded in
    /// the artifact can never disagree. Sorting and de-duplicating also fixes the
    /// order in which random factors are drawn, which is what makes `--seed`
    /// mean something.
    pub fn resolved(&self, layer_count: usize) -> Self {
        let mut layers = if self.layers.is_empty() {
            (0..layer_count).collect::<Vec<_>>()
        } else {
            self.layers.clone()
        };
        layers.sort_unstable();
        layers.dedup();
        Self {
            rank: self.rank,
            alpha: self.alpha,
            targets: self.targets.clone(),
            layers,
            seed: self.seed,
        }
    }

    /// The constant the low-rank product is multiplied by, `alpha / rank`.
    ///
    /// Dividing by the rank is what lets an operator raise the rank without also
    /// raising the effective learning rate of the update.
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}
