//! What a preference run is asked for, what it reports back, and what a
//! checkpoint of it carries.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::super::checkpoint::Durable;
use super::scoring::Summary;
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
    /// Where checkpoints are written and which one the run continues from.
    pub durable: Durable,
    /// What the run was started from beyond these options (model, revision,
    /// the digest of the pair set, encoding, precision); a checkpoint of a
    /// run with another fingerprint is refused.
    pub identity: BTreeMap<String, Value>,
}

impl DpoOptions {
    /// Everything a resumed run must share with the run that wrote the
    /// checkpoint.
    pub(super) fn fingerprint(&self) -> BTreeMap<String, Value> {
        let mut fingerprint = self.identity.clone();
        let targets: Vec<&str> = self.spec.targets.iter().map(|target| target.name()).collect();
        let options = [
            ("objective", Value::from("dpo")),
            ("loss", Value::from(self.loss.name())),
            ("beta", Value::from(self.beta)),
            ("rank", Value::from(self.spec.rank)),
            ("alpha", Value::from(self.spec.alpha)),
            ("targets", Value::from(targets)),
            ("layers", Value::from(self.spec.layers.clone())),
            ("epochs", Value::from(self.epochs)),
            ("learning_rate", Value::from(self.learning_rate)),
            ("accumulation", Value::from(self.accumulation)),
            ("batch", Value::from(self.batch)),
            ("warmup_steps", Value::from(self.warmup_steps)),
            ("max_sequence", Value::from(self.max_sequence)),
            ("seed", Value::from(self.seed)),
        ];
        fingerprint.extend(options.into_iter().map(|(field, value)| (field.to_owned(), value)));
        fingerprint
    }
}

/// What the loop keeps for its report, and what a checkpoint carries of it:
/// the first and latest step losses, and the totals of the epoch in progress
/// as [`Summary`] keeps them.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct Running {
    pub first_loss: Option<f32>,
    pub final_loss: Option<f32>,
    /// The mean loss of every step of the epoch in progress, and its totals.
    pub steps: Vec<f32>,
    pub epoch: Totals,
    /// The finished epochs, the last one's totals among them.
    pub finished: Vec<Totals>,
}

/// [`Summary`]'s totals, as a checkpoint writes them.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(super) struct Totals {
    pairs: usize,
    loss: f64,
    correct: usize,
    chosen_reward: f64,
    rejected_reward: f64,
}

impl From<&Summary> for Totals {
    fn from(summary: &Summary) -> Self {
        Self {
            pairs: summary.pairs,
            loss: summary.loss,
            correct: summary.correct,
            chosen_reward: summary.chosen_reward,
            rejected_reward: summary.rejected_reward,
        }
    }
}

impl From<&Totals> for Summary {
    fn from(totals: &Totals) -> Self {
        Self {
            pairs: totals.pairs,
            loss: totals.loss,
            correct: totals.correct,
            chosen_reward: totals.chosen_reward,
            rejected_reward: totals.rejected_reward,
        }
    }
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
    /// The step the run continued from, when it was resumed.
    pub resumed_from: Option<usize>,
    /// The steps after which a checkpoint was written.
    pub checkpoints: Vec<usize>,
}
