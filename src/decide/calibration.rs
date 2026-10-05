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

/// The uncalibrated temperature: logits used as the model produced them.
pub const RAW_TEMPERATURE: f64 = 1.0;

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
    /// The same labels scored at the fitted temperature with every question
    /// judged against another example's state. Present from two examples up.
    /// Reading the state means `after.accuracy` beats `control.accuracy`;
    /// equal means the answers came from the options and letters alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<Metrics>,
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
    /// Equal-width confidence bins the ECE was measured over, as the caller
    /// stated them; an ECE is only comparable with one taken over as many.
    /// Absent on calibrations written before the count was recorded.
    #[serde(default)]
    pub ece_bins: Option<usize>,
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
/// `labelled`, searched over its logarithm.
///
/// No window and no step count are assumed. The search starts at the raw
/// temperature and doubles its stride outward until the likelihood stops
/// improving, which brackets the minimum wherever it lies; the
/// negative log-likelihood of temperature scaling has one minimum, so the
/// bracket holds it. Golden-section search then narrows the bracket until
/// floating point cannot split it further. A likelihood that keeps improving
/// until `exp` leaves the finite range — every label already won with
/// certainty, or none ever can be — stops at the last finite temperature.
pub fn fit_temperature(labelled: &[Labelled]) -> f64 {
    let log_limit = f64::MAX.ln();
    let at = |log_temperature: f64| nll(labelled, log_temperature.exp());
    let origin = RAW_TEMPERATURE.ln();
    let (low, high) = bracket(&at, origin, log_limit);
    golden_section(&at, low, high).exp()
}

/// An interval of log temperatures holding the minimum: walks downhill from
/// `origin` with a doubling stride until the value rises again.
fn bracket(at: &impl Fn(f64) -> f64, origin: f64, log_limit: f64) -> (f64, f64) {
    let centre = at(origin);
    let mut stride = 1.0;
    // Downhill is whichever side improves; neither improving brackets at once.
    let direction = if at(origin + stride) < centre {
        1.0
    } else if at(origin - stride) < centre {
        -1.0
    } else {
        return (origin - stride, origin + stride);
    };
    let mut previous = origin;
    let mut current = origin + direction * stride;
    let mut value = at(current);
    loop {
        stride *= 2.0;
        let next = (current + direction * stride).clamp(-log_limit, log_limit);
        let next_value = at(next);
        if next_value >= value || next == current {
            let (a, b) = (previous, next);
            return if a < b { (a, b) } else { (b, a) };
        }
        previous = current;
        current = next;
        value = next_value;
    }
}

/// Golden-section search for the minimum of `at` on `[low, high]`, run until
/// the interval cannot be split into distinct floating-point probes.
fn golden_section(at: &impl Fn(f64) -> f64, mut low: f64, mut high: f64) -> f64 {
    let golden = (5f64.sqrt() - 1.0) / 2.0;
    let mut left = high - golden * (high - low);
    let mut right = low + golden * (high - low);
    let mut left_value = at(left);
    let mut right_value = at(right);
    while low < left && left < right && right < high {
        if left_value < right_value {
            high = right;
            right = left;
            right_value = left_value;
            left = high - golden * (high - low);
            left_value = at(left);
        } else {
            low = left;
            left = right;
            left_value = right_value;
            right = low + golden * (high - low);
            right_value = at(right);
        }
    }
    (low + high) / 2.0
}

fn nll(labelled: &[Labelled], temperature: f64) -> f64 {
    labelled
        .iter()
        .map(|item| -item.logits.probabilities(temperature)[item.truth].max(f64::MIN_POSITIVE).ln())
        .sum::<f64>()
        / labelled.len() as f64
}

/// Every metric at one temperature, the ECE over `ece_bins` equal-width bins.
pub fn metrics(labelled: &[Labelled], temperature: f64, ece_bins: usize) -> Metrics {
    let mut bins = vec![(0usize, 0f64, 0f64); ece_bins];
    let mut correct = 0usize;
    for item in labelled {
        let probabilities = item.logits.probabilities(temperature);
        let winner = argmax(&probabilities);
        let hit = winner == item.truth;
        correct += usize::from(hit);
        let top = probabilities[winner];
        let bin = ((top * ece_bins as f64) as usize).min(ece_bins - 1);
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
    Metrics { temperature, nll: nll(labelled, temperature), ece, ece_bins: Some(ece_bins), accuracy: correct as f64 / total }
}
