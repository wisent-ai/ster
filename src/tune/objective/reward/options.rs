//! What a reward run is asked for, what it reports back, and what a
//! checkpoint of it carries.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::super::checkpoint::Durable;
use super::step::Summary;
use crate::lora;

#[derive(Debug, Clone)]
pub struct RewardOptions {
    pub spec: lora::Spec,
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

impl RewardOptions {
    /// Everything a resumed run must share with the run that wrote the
    /// checkpoint.
    pub(super) fn fingerprint(&self) -> BTreeMap<String, Value> {
        let mut fingerprint = self.identity.clone();
        let targets: Vec<&str> = self
            .spec
            .targets
            .iter()
            .map(|target| target.name())
            .collect();
        let options = [
            ("objective", Value::from("reward")),
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
        fingerprint.extend(
            options
                .into_iter()
                .map(|(field, value)| (field.to_owned(), value)),
        );
        fingerprint
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RewardReport {
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
    /// The share of pairs the head already ranks correctly, over the final
    /// epoch — so it describes the head that was written, not an average over
    /// a head that was still moving.
    pub accuracy: f32,
    /// How many of those pairs the head scored exactly equal, over the same
    /// final epoch — the pairs `accuracy` counts as not-correct without them
    /// being wrong. A fresh head is zeros, so a run short enough that it never
    /// moved reports every pair tied here and 0.0 there; a head that learned
    /// the order backwards reports the same accuracy with no ties at all.
    pub tied_pairs: usize,
    pub mean_chosen_score: f32,
    pub mean_rejected_score: f32,
    pub mean_score_margin: f32,
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

/// What the loop keeps for its report, and what a checkpoint carries of it.
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
    tied: usize,
    chosen: f64,
    rejected: f64,
}

impl From<&Summary> for Totals {
    fn from(summary: &Summary) -> Self {
        Self {
            pairs: summary.pairs,
            loss: summary.loss,
            correct: summary.correct,
            tied: summary.tied,
            chosen: summary.chosen,
            rejected: summary.rejected,
        }
    }
}

impl From<&Totals> for Summary {
    fn from(totals: &Totals) -> Self {
        Self {
            pairs: totals.pairs,
            loss: totals.loss,
            correct: totals.correct,
            tied: totals.tied,
            chosen: totals.chosen,
            rejected: totals.rejected,
        }
    }
}
