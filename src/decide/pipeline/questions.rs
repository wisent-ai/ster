//! One labelled question of a benchmark on its own: the option its label
//! names, the option the model put the most mass on, and the probability of
//! every option, read unsteered and, when the run steers, steered as well. A
//! steered run counts the questions the steering turned right and the ones it
//! turned wrong from these rows.

use serde::Serialize;
use serde_json::Value;

use crate::{
    decide::{Question, QuestionLogits, answer::argmax},
    failures::{decide::pipeline::PREDICTION_UNNAMED, refuse},
};

/// One read of one question.
#[derive(Debug, Clone, Serialize)]
pub struct Reading {
    /// The option holding the most probability, the first one on a tie,
    /// written the way a label names it.
    pub predicted: Value,
    /// Whether `predicted` is the label's option.
    pub correct: bool,
    /// One probability per option, in the order of the row's `options`.
    pub probabilities: Vec<f64>,
}

/// One labelled question as the benchmark read it.
#[derive(Debug, Clone, Serialize)]
pub struct QuestionRow {
    /// The example's position in the set, counted from zero.
    pub example: usize,
    /// The question's id.
    pub question: String,
    /// Every option as a label names it, in canonical order: a choice's option
    /// by its name, a score's level by its index, a noul's `true` then
    /// `false`.
    pub options: Vec<Value>,
    /// The label.
    pub gold: Value,
    /// The read with nothing steered.
    pub base: Reading,
    /// The read under the run's steering artifacts; absent from an
    /// unsteered run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steered: Option<Reading>,
}

impl Reading {
    /// What `logits` say about question `question` of example `example`,
    /// whose options are `options` and whose label is option `truth`, at
    /// `temperature`; refused as `prediction-unnamed` when the option the
    /// probabilities pick is not one of `options`.
    pub(super) fn new(
        example: usize,
        question: &str,
        options: &[Value],
        logits: &QuestionLogits,
        truth: usize,
        temperature: f64,
    ) -> anyhow::Result<Self> {
        let probabilities = logits.probabilities(temperature);
        let picked = argmax(&probabilities);
        let Some(predicted) = options.get(picked) else {
            return Err(refuse(
                &PREDICTION_UNNAMED,
                &[
                    ("question", question),
                    ("example", &example.to_string()),
                    ("picked", &picked.to_string()),
                    ("count", &options.len().to_string()),
                    ("probabilities", &format!("{probabilities:?}")),
                ],
            ));
        };
        Ok(Self {
            predicted: predicted.clone(),
            correct: picked == truth,
            probabilities,
        })
    }
}

/// `question`'s options as its labels name them, in the canonical order its
/// logits are kept in (`Question::options`, `Question::truth_index`).
pub(super) fn option_values(question: &Question) -> Vec<Value> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .keys()
            .map(|name| Value::String(name.clone()))
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(level, _)| Value::from(level))
            .collect(),
        Question::Noul { .. } => vec![Value::Bool(true), Value::Bool(false)],
    }
}

/// How many questions the steering turned from wrong to right, and how many
/// from right to wrong; both zero for rows read without steering.
pub(super) fn changed(rows: &[QuestionRow]) -> (usize, usize) {
    let mut improved: usize = Default::default();
    let mut worsened: usize = Default::default();
    for row in rows {
        match (row.base.correct, row.steered.as_ref().map(|steered| steered.correct)) {
            (false, Some(true)) => improved += 1,
            (true, Some(false)) => worsened += 1,
            _ => {}
        }
    }
    (improved, worsened)
}
