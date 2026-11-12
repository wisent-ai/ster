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

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::failures::{
    lora::{
        ALPHA_UNUSABLE, LAYER_PAST, LAYERS_NONE, RANK_ZERO, TARGET_TWICE, TARGET_UNKNOWN,
        TARGETS_NONE,
    },
    refuse,
};

mod adapter;
mod artifact;

pub use adapter::{Adapter, Adapters};
pub use artifact::{ARTIFACT_SCHEMA_VERSION, Artifact, Kind, REWARD_HEAD_TENSOR};

/// The widths a model's projections have, which an adapter's factors must
/// match: the residual stream, the query projection, the attention output
/// (narrower than the query when a head's value is narrower than its query,
/// as in DeepSeek's latent attention), one key or value projection, and the
/// feed-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widths {
    pub hidden: usize,
    pub attention: usize,
    pub attention_output: usize,
    pub key_value: usize,
    pub intermediate: usize,
}

impl Widths {
    /// Projection dimensions of the shared causal decoder, including unequal head widths.
    pub(crate) fn for_decoder(
        config: &candle_transformers::models::llama::Config,
        architecture: &crate::model::Architecture,
    ) -> Self {
        Self {
            hidden: config.hidden_size,
            attention: architecture.attention_width(config.num_attention_heads),
            attention_output: config.num_attention_heads * architecture.value_dim(),
            key_value: config.num_key_value_heads * architecture.head_dim,
            intermediate: config.intermediate_size,
        }
    }
}

/// Which projections carry adapters.
///
/// `Ord` is derived because [`Adapters`] keys a [`BTreeMap`] by
/// `(layer, target)`, which is what makes adapter construction — and therefore
/// the sequence of draws from the random initialiser — reproducible for a seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
            _ => Err(refuse(
                &TARGET_UNKNOWN,
                &[("target", &format!("{text:?}")), ("targets", text)],
            )),
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
            Self::Output => (widths.hidden, widths.attention_output),
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
    /// Query writes the attention width, which need not equal the residual width.
    fn writes_hidden(self) -> bool {
        matches!(self, Self::Output | Self::Down)
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
    pub fn check(&self) -> Result<()> {
        if self.rank == 0 {
            return Err(refuse(&RANK_ZERO, &[("rank", &self.rank.to_string())]));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(refuse(&ALPHA_UNUSABLE, &[("alpha", &self.alpha.to_string())]));
        }
        let named: Vec<&str> = self.targets.iter().map(|target| target.name()).collect();
        if self.targets.is_empty() {
            return Err(refuse(&TARGETS_NONE, &[("targets", "\"\"")]));
        }
        for (index, target) in self.targets.iter().enumerate() {
            if self.targets[..index].contains(target) {
                return Err(refuse(
                    &TARGET_TWICE,
                    &[("target", target.name()), ("targets", &named.join(","))],
                ));
            }
        }
        Ok(())
    }

    /// [`Self::check`], a model with at least one layer, and every layer the
    /// spec names within the model's `layer_count`.
    pub fn validate(&self, layer_count: usize) -> Result<()> {
        self.check()?;
        if layer_count == 0 {
            return Err(refuse(&LAYERS_NONE, &[("count", "0")]));
        }
        if let Some(layer) = self.layers.iter().find(|layer| **layer >= layer_count) {
            let listed: Vec<String> = self.layers.iter().map(usize::to_string).collect();
            return Err(refuse(
                &LAYER_PAST,
                &[
                    ("layer", &layer.to_string()),
                    ("count", &layer_count.to_string()),
                    ("last", &layer_count.saturating_sub(1).to_string()),
                    ("layers", &listed.join(",")),
                ],
            ));
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
