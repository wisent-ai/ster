//! What a BiPO run is asked for, what it reports, the refusals of settings
//! no run can train with, and the fingerprint a checkpoint of it carries.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::Serialize;
use serde_json::Value;

use super::super::super::super::checkpoint::Durable;
use super::super::DpoLoss;

/// Every setting one BiPO run takes; Ster assumes none of them.
#[derive(Debug, Clone)]
pub struct BipoOptions {
    /// The layer whose residual stream the vector is added to.
    pub layer: usize,
    /// The multiple of `v` added while training, in each direction.
    pub strength: f64,
    pub loss: DpoLoss,
    pub beta: f64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub accumulation: usize,
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

#[derive(Debug, Clone, Serialize)]
pub struct BipoReport {
    pub layer: usize,
    pub strength: f64,
    pub loss: String,
    pub beta: f64,
    pub pairs: usize,
    pub trained_pairs: usize,
    pub skipped_long: usize,
    pub epochs: usize,
    pub steps: usize,
    pub first_loss: f32,
    pub final_loss: f32,
    pub mean_final_epoch_loss: f32,
    pub accuracy: f32,
    pub mean_reward_margin: f32,
    pub vector_norm: f32,
    pub learning_rate: f64,
    pub accumulation: usize,
    pub batch: usize,
    /// The step the run continued from, when it was resumed.
    pub resumed_from: Option<usize>,
    /// The steps after which a checkpoint was written.
    pub checkpoints: Vec<usize>,
}

impl BipoOptions {
    /// Refuses a run whose settings cannot train: each refusal names its
    /// setting.
    pub(super) fn validate(&self, layers: usize) -> Result<()> {
        if self.layer >= layers {
            bail!(
                "layer {} is outside the model's {layers} layers",
                self.layer
            );
        }
        for (name, value) in [
            ("beta", self.beta),
            ("strength", self.strength),
            ("learning rate", self.learning_rate),
        ] {
            // A normal, positive float: finite, not zero, not subnormal.
            if !(value.is_normal() && value.is_sign_positive()) {
                bail!(
                    "bi-directional preference optimization requires a finite {name} above zero, not {value}"
                );
            }
        }
        for (name, value) in [
            ("epochs", self.epochs),
            ("accumulation", self.accumulation),
            ("batch size", self.batch),
            ("sequence limit", self.max_sequence),
        ] {
            if value == 0 {
                bail!("bi-directional preference optimization requires {name} of at least one");
            }
        }
        Ok(())
    }

    /// Everything a resumed run must share with the run that wrote the
    /// checkpoint.
    pub(super) fn fingerprint(&self) -> BTreeMap<String, Value> {
        let mut fingerprint = self.identity.clone();
        let options = [
            ("objective", Value::from("bipo")),
            ("layer", Value::from(self.layer)),
            ("strength", Value::from(self.strength)),
            ("loss", Value::from(self.loss.name())),
            ("beta", Value::from(self.beta)),
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
