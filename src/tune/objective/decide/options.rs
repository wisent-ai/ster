//! What a calibrated decision training run is given and what it reports,
//! what a checkpoint of it carries, and its fingerprint.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::super::checkpoint::Durable;
use crate::lora;

#[derive(Debug, Clone)]
pub struct DecideOptions {
    pub spec: lora::Spec,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
    /// Rows folded into one forward pass.
    pub batch: usize,
    pub warmup_steps: usize,
    pub max_sequence: usize,
    /// Option orders per question: `0` for every cyclic shift.
    pub permutations: usize,
    pub seed: u64,
    /// Where checkpoints are written and which one the run continues from.
    pub durable: Durable,
    /// What the run was started from, beyond these options: the model and
    /// its revision, the digest of the labelled decisions, the encoding and
    /// precision. A checkpoint of a run with another fingerprint is refused.
    pub identity: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecideReport {
    pub examples: usize,
    /// Labelled questions across the examples.
    pub questions: usize,
    /// Training rows: one per labelled question per option order.
    pub rows: usize,
    pub trained_rows: usize,
    pub skipped_long: usize,
    pub epochs: usize,
    pub steps: usize,
    pub trainable_tensors: usize,
    pub trainable_parameters: usize,
    /// Mean negative log-probability of the correct option, at the first
    /// step and the last, and over the final epoch.
    pub first_loss: f32,
    pub final_loss: f32,
    pub mean_final_epoch_loss: f32,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub layers: Vec<usize>,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
    pub permutations: usize,
    /// The step the run continued from, when it was resumed.
    pub resumed_from: Option<usize>,
    /// The steps after which a checkpoint was written.
    pub checkpoints: Vec<usize>,
}

/// What the loop keeps for its report, and what a checkpoint carries of it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct Running {
    pub first_loss: Option<f32>,
    pub final_loss: Option<f32>,
    /// The summed loss and the rows of every step of the current epoch.
    pub epoch_losses: Vec<f64>,
    pub epoch_units: Vec<usize>,
    /// The mean loss of every finished epoch.
    pub epoch_means: Vec<f32>,
}

impl DecideOptions {
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
            ("objective", Value::from("decide")),
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
            ("permutations", Value::from(self.permutations)),
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
