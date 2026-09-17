//! Making the probabilities honest: one temperature, fitted on labelled
//! decisions, that the answer-letter logits are divided by before the
//! softmax. A model that is right nine times in ten should say `0.9`, and a
//! raw language model rarely does — it is usually too sure. Temperature
//! scaling is the smallest correction that fixes that: one number, fitted by
//! minimizing the negative log-likelihood of the labels, which cannot change
//! which option wins and so cannot trade accuracy for calibration.

use std::{fs, path::Path};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::answer::{argmax, QuestionLogits};

pub const SCHEMA: &str = "ster-calibration/1";

/// The search window for the fitted temperature, as bounds on its logarithm.
/// Twenty times sharper than raw and twenty times flatter cover every model
/// this has been run on with room to spare; a fit that lands on a bound is
/// reported as the bound, which an operator can read as a warning.
const LOG_TEMPERATURE_MIN: f64 = -3.0;
const LOG_TEMPERATURE_MAX: f64 = 3.0;

/// Golden-section steps over that window. Each step shrinks it by 0.618, so
/// sixty of them resolve the temperature to far below the noise of any
/// labelled set an operator will actually have.
const SEARCH_STEPS: usize = 60;

/// The uncalibrated temperature: logits used as the model produced them.
pub const RAW_TEMPERATURE: f64 = 1.0;

/// Reliability bins for the expected calibration error, the number the
/// calibration literature reports with and the one a reader can compare
/// against.
const ECE_BINS: usize = 10;

/// The artifact `ster calibrate` writes and `ster decide --calibration`
/// reads. It names the model it was fitted on, because a temperature fitted
/// on one checkpoint says nothing about another.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Calibration {
    pub schema: String,
    pub model: String,
    pub revision: Option<String>,
    pub chat_template: String,
    pub precision: String,
    pub temperature: f64,
    /// Labelled examples the fit read.
    pub examples: usize,
    /// Labelled questions across them: the sample the metrics are over.
    pub questions: usize,
    pub before: Metrics,
    pub after: Metrics,
}

/// How well the probabilities matched the labels, at one temperature.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Metrics {
    pub temperature: f64,
    /// Mean negative log-likelihood of the correct option.
    pub nll: f64,
    /// Expected calibration error: how far the confidence of the winning
    /// option is, on average, from how often it is right.
    pub ece: f64,
    /// How often the winning option was the labelled one. The same at every
    /// temperature, because scaling never changes the winner.
    pub accuracy: f64,
}

impl Calibration {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let calibration: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid calibration {}", path.display()))?;
        if calibration.schema != SCHEMA {
            bail!(
                "calibration {} has schema {:?}; this Ster reads {SCHEMA}",
                path.display(),
                calibration.schema
            );
        }
        if !(calibration.temperature.is_finite() && calibration.temperature > 0.0) {
            bail!("calibration {} has a temperature that is not a positive number", path.display());
        }
        Ok(calibration)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    /// Refuses a calibration fitted on a different checkpoint than the one
    /// about to use it.
    pub fn check_model(&self, path: &Path, model: &str) -> Result<()> {
        if self.model != model {
            bail!(
                "calibration {} was fitted for model '{}', not '{model}'",
                path.display(),
                self.model
            );
        }
        Ok(())
    }
}

/// One labelled question: its logits and which canonical option is right.
#[derive(Debug, Clone)]
pub struct Labelled {
    pub logits: QuestionLogits,
    pub truth: usize,
}

/// The temperature that minimizes the mean negative log-likelihood over
/// `labelled`, by golden-section search over its logarithm.
pub fn fit_temperature(labelled: &[Labelled]) -> f64 {
    let golden = (5f64.sqrt() - 1.0) / 2.0;
    let (mut low, mut high) = (LOG_TEMPERATURE_MIN, LOG_TEMPERATURE_MAX);
    let mut left = high - golden * (high - low);
    let mut right = low + golden * (high - low);
    let mut left_value = nll(labelled, left.exp());
    let mut right_value = nll(labelled, right.exp());
    for _ in 0..SEARCH_STEPS {
        if left_value < right_value {
            high = right;
            right = left;
            right_value = left_value;
            left = high - golden * (high - low);
            left_value = nll(labelled, left.exp());
        } else {
            low = left;
            left = right;
            left_value = right_value;
            right = low + golden * (high - low);
            right_value = nll(labelled, right.exp());
        }
    }
    ((low + high) / 2.0).exp()
}

fn nll(labelled: &[Labelled], temperature: f64) -> f64 {
    labelled
        .iter()
        .map(|item| -item.logits.probabilities(temperature)[item.truth].max(f64::MIN_POSITIVE).ln())
        .sum::<f64>()
        / labelled.len() as f64
}

/// Every metric at one temperature.
pub fn metrics(labelled: &[Labelled], temperature: f64) -> Metrics {
    let mut bins = vec![(0usize, 0f64, 0f64); ECE_BINS];
    let mut correct = 0usize;
    for item in labelled {
        let probabilities = item.logits.probabilities(temperature);
        let winner = argmax(&probabilities);
        let hit = winner == item.truth;
        correct += usize::from(hit);
        let top = probabilities[winner];
        let bin = ((top * ECE_BINS as f64) as usize).min(ECE_BINS - 1);
        let (count, confidence_sum, hit_sum) = &mut bins[bin];
        *count += 1;
        *confidence_sum += top;
        *hit_sum += f64::from(u8::from(hit));
    }
    let total = labelled.len() as f64;
    let ece = bins
        .iter()
        .filter(|(count, _, _)| *count > 0)
        .map(|(count, confidence_sum, hit_sum)| {
            let count = *count as f64;
            (count / total) * (confidence_sum / count - hit_sum / count).abs()
        })
        .sum();
    Metrics { temperature, nll: nll(labelled, temperature), ece, accuracy: correct as f64 / total }
}
