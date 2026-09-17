//! What a decision returns, and the arithmetic that turns answer-letter
//! logits into it.

use std::collections::BTreeMap;

use serde::Serialize;

use super::{
    calibration::RAW_TEMPERATURE,
    prompt::LABELS,
    request::{text_of, Question},
};

/// The confidence of a distribution with nothing to be uncertain between: a
/// single option holds all the mass by construction.
const CERTAIN: f64 = 1.0;

/// One answer, tagged like the question that produced it.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice { choice: String, probabilities: BTreeMap<String, f64>, confidence: f64 },
    Score {
        score: f64,
        legend: BTreeMap<usize, String>,
        probabilities: BTreeMap<usize, f64>,
        confidence: f64,
    },
    Noul { noul: f64 },
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Prompt tokens over every rendering of every question.
    pub input_tokens: usize,
    /// Sequences the model ran: one per question per option order.
    pub forward_passes: usize,
    /// Always zero. A decision is read from the distribution at one position
    /// and no token is ever generated.
    pub output_tokens: usize,
}

/// The tokens a decision generates: none. A decision is read from the
/// distribution at one position, and the field exists so a reader of the
/// TypeSafe-shaped usage block sees the number rather than a missing key.
const GENERATED_TOKENS: usize = 0;

impl Usage {
    /// The cost of `forward_passes` sequences before any of them has run;
    /// `input_tokens` is added as each row is read.
    pub fn for_passes(forward_passes: usize) -> Self {
        Self { input_tokens: 0, forward_passes, output_tokens: GENERATED_TOKENS }
    }
}

/// The document `ster decide` prints.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub model: String,
    pub revision: Option<String>,
    pub chat_template: &'static str,
    pub precision: &'static str,
    /// The temperature the answer-letter logits were divided by: `1.0`
    /// uncalibrated, or the fitted value from `calibration`.
    pub temperature: f64,
    pub calibration: Option<String>,
    /// The operator's option-order setting, `0` meaning every cyclic shift.
    pub permutations: usize,
    pub answers: BTreeMap<String, Answer>,
    /// Per-order detail for every question, present under `--explain`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<BTreeMap<String, Explanation>>,
    pub usage: Usage,
}

/// How one answer came about: the option texts in canonical order, and what
/// the model said in each order it was shown them in.
#[derive(Debug, Clone, Serialize)]
pub struct Explanation {
    pub options: Vec<String>,
    pub orders: Vec<Order>,
}

/// One order: the letter each canonical option wore, and the probability the
/// model put on it in that order alone.
#[derive(Debug, Clone, Serialize)]
pub struct Order {
    pub letters: Vec<&'static str>,
    pub probabilities: Vec<f64>,
}

impl Explanation {
    pub fn new(options: Vec<String>, logits: &QuestionLogits) -> Self {
        let orders = logits
            .orders
            .iter()
            .zip(logits.per_order())
            .map(|(order, probabilities)| Order {
                letters: (0..order.len())
                    .map(|option| LABELS[order.iter().position(|&shown| shown == option).expect("every option is shown once")])
                    .collect(),
                probabilities,
            })
            .collect();
        Self { options, orders }
    }
}

/// The answer-letter logits one question produced: one row per option order,
/// each row in canonical option order whatever letters the options wore.
#[derive(Debug, Clone, Default)]
pub struct QuestionLogits {
    pub rows: Vec<Vec<f32>>,
    /// The option order each row was rendered in: position to canonical
    /// option.
    pub orders: Vec<Vec<usize>>,
}

impl QuestionLogits {
    /// One log-probability per option: each row normalized on its own, then
    /// averaged across the orders.
    ///
    /// Normalizing per row first is what removes the letter preference. A
    /// model that favours the second letter gives that letter's option most
    /// of the mass in every order; across all cyclic orders every option
    /// wears that letter once, so the favour lands on each of them equally
    /// and cancels in the mean, while whatever the model reads from the
    /// content survives in every order and adds up. The mean is taken over
    /// log-probabilities rather than probabilities because one order with a
    /// favoured letter would otherwise outvote the rest by arithmetic alone.
    pub fn log_scores(&self) -> Vec<f64> {
        let count = self.rows.first().map_or(0, Vec::len);
        let mut mean = vec![0f64; count];
        for row in &self.rows {
            for (slot, value) in mean.iter_mut().zip(log_softmax(row)) {
                *slot += value / self.rows.len() as f64;
            }
        }
        mean
    }

    /// The probability of each option: the averaged log-scores, divided by
    /// `temperature`, through one softmax. Temperature scales the whole
    /// distribution's sharpness and never changes which option wins.
    pub fn probabilities(&self, temperature: f64) -> Vec<f64> {
        let scores: Vec<f32> = self.log_scores().into_iter().map(|value| value as f32).collect();
        softmax(&scores, temperature)
    }

    /// What the model said in each order on its own: the per-order softmax in
    /// canonical option order, which is what an operator reads to see how
    /// much of an answer was the letter and how much was the content.
    pub fn per_order(&self) -> Vec<Vec<f64>> {
        self.rows.iter().map(|row| softmax(row, RAW_TEMPERATURE)).collect()
    }
}

pub fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    log_softmax_at(logits, temperature).into_iter().map(f64::exp).collect()
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    log_softmax_at(logits, RAW_TEMPERATURE)
}

fn log_softmax_at(logits: &[f32], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|&value| f64::from(value) / temperature).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = scaled.iter().map(|value| (value - max).exp()).sum();
    let normalizer = max + total.ln();
    scaled.into_iter().map(|value| value - normalizer).collect()
}

/// `log(sum(exp(values)))`, for summing the mass a letter holds across its
/// spellings without leaving the log domain.
pub fn log_sum_exp(values: impl Iterator<Item = f32>) -> f32 {
    let values: Vec<f32> = values.collect();
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max + values.iter().map(|value| (value - max).exp()).sum::<f32>().ln()
}

/// How peaked a distribution is, from `0` (uniform) to `1` (all on one
/// option): one minus the entropy as a fraction of the most it could be.
pub fn confidence(probabilities: &[f64]) -> f64 {
    let count = probabilities.len();
    if count < 2 {
        return CERTAIN;
    }
    let entropy: f64 = probabilities
        .iter()
        .filter(|&&probability| probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum();
    (CERTAIN - entropy / (count as f64).ln()).clamp(0.0, CERTAIN)
}

/// The index of the largest probability, the first one on a tie.
pub fn argmax(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .fold(0, |best, (index, &value)| if value > probabilities[best] { index } else { best })
}

impl Answer {
    /// The typed answer for `question` from its canonical-order probabilities.
    pub fn from_probabilities(question: &Question, probabilities: Vec<f64>) -> Self {
        match question {
            Question::Choice { criteria, .. } => {
                let names: Vec<&String> = criteria.keys().collect();
                let choice = names[argmax(&probabilities)].clone();
                let confidence = confidence(&probabilities);
                let probabilities = names.into_iter().cloned().zip(probabilities).collect();
                Self::Choice { choice, probabilities, confidence }
            }
            Question::Score { criteria, .. } => {
                let score = probabilities
                    .iter()
                    .enumerate()
                    .map(|(level, probability)| level as f64 * probability)
                    .sum();
                let confidence = confidence(&probabilities);
                let legend = criteria
                    .iter()
                    .enumerate()
                    .map(|(level, description)| (level, text_of(description)))
                    .collect();
                let probabilities = probabilities.into_iter().enumerate().collect();
                Self::Score { score, legend, probabilities, confidence }
            }
            Question::Noul { .. } => Self::Noul { noul: probabilities[0] },
        }
    }
}
