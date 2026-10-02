//! What a preference run is asked for, and what it reports back.

use anyhow::{Result, bail};
use serde::Serialize;

use crate::lora;

/// Which preference objective the log-ratio margin is fed into.
///
/// Both read the same pair set, take the same forward passes, and differ only
/// in the scalar function applied at the very end, which is why supporting the
/// second costs one match arm rather than a second trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpoLoss {
    Dpo,
    Ipo,
}

impl DpoLoss {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dpo" => Ok(Self::Dpo),
            "ipo" => Ok(Self::Ipo),
            _ => bail!("unknown preference loss {value:?}; expected dpo or ipo"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Dpo => "dpo",
            Self::Ipo => "ipo",
        }
    }

    /// Whether the log-probabilities are divided by the number of tokens they
    /// sum over.
    ///
    /// DPO's derivation is over whole-sequence log-probabilities and length
    /// enters the objective on purpose. IPO replaces the sigmoid with a squared
    /// error against a fixed target, and a target that a long sequence reaches
    /// by length alone is not a preference signal, so IPO scores the mean
    /// per-token log-probability instead.
    pub(super) fn length_normalized(self) -> bool {
        matches!(self, Self::Ipo)
    }
}

#[derive(Debug, Clone)]
pub struct DpoOptions {
    pub spec: lora::Spec,
    pub loss: DpoLoss,
    /// How hard the reference pulls back. Small beta lets the policy move far
    /// from the reference before the loss objects; the 0.1 default is the value
    /// the DPO paper reports across its settings.
    pub beta: f64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
    /// Pairs folded into one forward pass. A pair is two rows, so a batch of
    /// four pairs is a forward of eight sequences; one is the unbatched pass
    /// every run recorded before batching existed.
    pub batch: usize,
    pub warmup_steps: usize,
    pub max_sequence: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DpoReport {
    pub loss: String,
    pub beta: f64,
    pub pairs: usize,
    pub trained_pairs: usize,
    pub skipped_long: usize,
    pub epochs: usize,
    pub steps: usize,
    pub trainable_tensors: usize,
    pub trainable_parameters: usize,
    pub first_loss: f32,
    pub final_loss: f32,
    pub mean_final_epoch_loss: f32,
    /// The share of pairs the policy already prefers the chosen side on,
    /// measured over the final epoch — so it describes the adapter that was
    /// written, not an average over a policy that was still moving.
    pub accuracy: f32,
    pub mean_reward_margin: f32,
    pub mean_chosen_reward: f32,
    pub mean_rejected_reward: f32,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub layers: Vec<usize>,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
}
