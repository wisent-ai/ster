//! What a BiPO run is asked for, what it reports, the refusals of settings
//! no run can train with, and the fingerprint a checkpoint of it carries.

use std::collections::BTreeMap;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use super::super::super::super::{checkpoint::Durable, preflight::Preflight};
use super::super::DpoLoss;
use crate::failures::{
    refuse,
    tune::objectives::{BETA_UNUSABLE, LAYER_OUTSIDE, STRENGTH_UNUSABLE},
};

/// How BiPO names itself in what it says.
pub(super) const SUBJECT: &str = "bi-directional preference optimization";

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
    /// Refuses the settings no BiPO run can train with, before any model or
    /// data is read. Beta and strength must be normal positive floats:
    /// finite, not zero, not subnormal.
    pub fn check(&self) -> Result<()> {
        Preflight {
            subject: SUBJECT,
            unit: "pair",
            pass: "epoch",
            noun: "steering vector",
            epochs: self.epochs,
            accumulation: self.accumulation,
            batch: Some(self.batch),
            learning_rate: self.learning_rate,
            max_sequence: self.max_sequence,
        }
        .check()?;
        let usable = |value: f64| value.is_normal() && value.is_sign_positive();
        if !usable(self.beta) {
            return Err(refuse(
                &BETA_UNUSABLE,
                &[("subject", SUBJECT), ("value", &self.beta.to_string())],
            ));
        }
        if !usable(self.strength) {
            return Err(refuse(
                &STRENGTH_UNUSABLE,
                &[("value", &self.strength.to_string())],
            ));
        }
        Ok(())
    }

    /// Refuses a layer the loaded model, of `layers` layers, does not have.
    pub(super) fn check_layer(&self, layers: usize) -> Result<()> {
        if self.layer < layers {
            return Ok(());
        }
        Err(refuse(
            &LAYER_OUTSIDE,
            &[
                ("layer", &self.layer.to_string()),
                ("layers", &layers.to_string()),
                ("last", &layers.saturating_sub(1).to_string()),
            ],
        ))
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
