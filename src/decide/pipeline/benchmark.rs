//! Measuring a model on a held-out labelled set: how often it is right, how
//! honest its probabilities are, whether it read the state, and how long a
//! decision takes. One document per run, so a base model and the adapter
//! trained on it are compared on the same rows with the same numbers, and a
//! run that steers reads every question both ways, so the base model and the
//! steered one are too.

use std::{collections::BTreeMap, time::Instant};

use anyhow::Result;
use serde::Serialize;

use super::questions::{QuestionRow, Reading, changed, option_values};
use crate::{
    decide::{
        ExampleSet, Labelled, Options, Question, Request, Steered, Steering,
        calibration::{Metrics, metrics},
        control_state, question_logits,
    },
    runtime::Runtime,
    workflow,
};

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub read: Options,
    /// The calibration artifact the temperature came from, if any.
    pub calibration: Option<String>,
    /// The adapter the runtime was loaded with, if any.
    pub adapter: Option<String>,
    /// Equal-width confidence bins the ECE is measured over, as the caller stated them.
    pub ece_bins: usize,
    /// The steering artifacts every question is read under as well, if any.
    pub steering: Option<Steering>,
}

/// The document `ster decisions benchmark` writes.
#[derive(Debug, Clone, Serialize)]
pub struct Benchmark {
    pub model: String,
    pub revision: Option<String>,
    pub adapter: Option<String>,
    pub chat_template: String,
    pub precision: String,
    pub calibration: Option<String>,
    pub permutations: usize,
    pub examples: usize,
    pub questions: usize,
    /// Every labelled question at the temperature the run used, unsteered.
    pub metrics: Metrics,
    /// The same labels with each question judged against another example's
    /// state; absent below two examples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control: Option<Metrics>,
    /// Per question type.
    pub by_type: BTreeMap<String, TypeMetrics>,
    /// The same questions read under the steering artifacts; absent from an
    /// unsteered run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steered: Option<SteeredReport>,
    /// The unsteered reads.
    pub latency: Latency,
    /// Tokens the model processed across every example, prefix and
    /// suffixes, in the unsteered reads.
    pub input_tokens: usize,
    /// Every labelled question on its own, in the set's order.
    pub rows: Vec<QuestionRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeMetrics {
    pub questions: usize,
    pub accuracy: f64,
    pub nll: f64,
}

/// A steered run's measure of the steering: the same numbers as the
/// unsteered `metrics` and `by_type`, and how many questions it changed.
#[derive(Debug, Clone, Serialize)]
pub struct SteeredReport {
    /// Each artifact added, with its strength.
    pub vectors: Vec<Steered>,
    pub metrics: Metrics,
    pub by_type: BTreeMap<String, TypeMetrics>,
    /// Questions the base model got wrong and the steered model right.
    pub improved: usize,
    /// Questions the base model got right and the steered model wrong.
    pub worsened: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Latency {
    /// Wall time over every example, model already loaded.
    pub total_seconds: f64,
    pub mean_ms_per_example: f64,
    pub mean_ms_per_question: f64,
}

const MILLISECONDS: f64 = 1000.0;

/// Reads every example of `set` and scores its labels, unsteered and, when
/// `options` carries steering, steered.
pub fn benchmark(
    runtime: &Runtime,
    set: &ExampleSet,
    options: &BenchmarkOptions,
) -> Result<Benchmark> {
    super::super::check_bins(options.ece_bins)?;
    set.validate()?;
    let count = set.examples.len();
    let temperature = options.read.temperature;
    let plan = options.steering.as_ref().map(|steering| &steering.plan);
    let mut labelled: Vec<(String, Labelled)> = Vec::new();
    let mut steered_labelled: Vec<(String, Labelled)> = Vec::new();
    let mut shuffled: Vec<Labelled> = Vec::new();
    let mut rows: Vec<QuestionRow> = Vec::new();
    let mut input_tokens = 0usize;
    let mut timed = 0f64;
    for (index, example) in set.examples.iter().enumerate() {
        let number = index + 1;
        workflow::progress(format!("benchmarking example {number} of {count}"));
        let request = labelled_request(&example.request, &example.answers);
        let clock = Instant::now();
        let (logits, usage) = question_logits(runtime, &request, options.read, None)?;
        timed += clock.elapsed().as_secs_f64();
        input_tokens += usage.input_tokens;
        let steered = match plan {
            Some(plan) => question_logits(runtime, &request, options.read, Some(plan))?.0,
            None => Vec::new(),
        };
        let mut steered = steered.into_iter();
        for ((id, question), logits) in request.questions.iter().zip(logits) {
            let gold = &example.answers[id];
            let truth = question.truth_index(gold).expect("validated label");
            let shown = option_values(question);
            let steered_logits = steered.next();
            rows.push(QuestionRow {
                example: index,
                question: id.clone(),
                gold: gold.clone(),
                base: Reading::new(index, id, &shown, &logits, truth, temperature)?,
                steered: steered_logits
                    .as_ref()
                    .map(|logits| Reading::new(index, id, &shown, logits, truth, temperature))
                    .transpose()?,
                options: shown,
            });
            if let Some(logits) = steered_logits {
                steered_labelled.push((kind(question).to_owned(), Labelled { logits, truth }));
            }
            labelled.push((kind(question).to_owned(), Labelled { logits, truth }));
        }
        if count > 1 {
            for (id, question) in &request.questions {
                let control = Request {
                    state: control_state(set, index, id).clone(),
                    model: None,
                    questions: BTreeMap::from([(id.clone(), question.clone())]),
                };
                let (logits, _) = question_logits(runtime, &control, options.read, None)?;
                let truth = question
                    .truth_index(&example.answers[id])
                    .expect("validated label");
                shuffled.push(Labelled {
                    logits: logits.into_iter().next().expect("one question"),
                    truth,
                });
            }
        }
    }
    let all = items(&labelled);
    let by_type = per_type(&labelled, temperature, options.ece_bins);
    let overall = metrics(&all, temperature, options.ece_bins);
    let control = (count > 1).then(|| metrics(&shuffled, temperature, options.ece_bins));
    workflow::progress(format!(
        "{} labelled questions: accuracy {:.4}, nll {:.4}, ece {:.4}{}; {:.0} ms per example",
        all.len(),
        overall.accuracy,
        overall.nll,
        overall.ece,
        match &control {
            Some(control) => format!(", {:.4} on shuffled states", control.accuracy),
            None => ", no shuffled-state control with one example".to_owned(),
        },
        timed * MILLISECONDS / count as f64
    ));
    let steered = options.steering.as_ref().map(|steering| {
        let (improved, worsened) = changed(&rows);
        let report = SteeredReport {
            vectors: steering.vectors.clone(),
            metrics: metrics(&items(&steered_labelled), temperature, options.ece_bins),
            by_type: per_type(&steered_labelled, temperature, options.ece_bins),
            improved,
            worsened,
        };
        workflow::progress(format!(
            "steered: accuracy {:.4} against {:.4} unsteered, {improved} turned right, {worsened} turned wrong",
            report.metrics.accuracy,
            overall.accuracy
        ));
        report
    });
    Ok(Benchmark {
        model: runtime.model_id.clone(),
        revision: runtime.revision.clone(),
        adapter: options.adapter.clone(),
        chat_template: runtime.chat_status().label().to_owned(),
        precision: runtime.precision().name().to_owned(),
        calibration: options.calibration.clone(),
        permutations: options.read.permutations,
        examples: count,
        questions: all.len(),
        metrics: overall,
        control,
        by_type,
        steered,
        latency: Latency {
            total_seconds: timed,
            mean_ms_per_example: timed * MILLISECONDS / count as f64,
            mean_ms_per_question: timed * MILLISECONDS / all.len().max(1) as f64,
        },
        input_tokens,
        rows,
    })
}

/// The labelled questions of `labelled`, their types left behind.
fn items(labelled: &[(String, Labelled)]) -> Vec<Labelled> {
    labelled.iter().map(|(_, item)| item.clone()).collect()
}

/// The accuracy and likelihood of `labelled` per question type, for every
/// type it holds.
fn per_type(
    labelled: &[(String, Labelled)],
    temperature: f64,
    ece_bins: usize,
) -> BTreeMap<String, TypeMetrics> {
    let mut by_type = BTreeMap::new();
    for name in ["choice", "score", "noul"] {
        let items: Vec<Labelled> = labelled
            .iter()
            .filter(|(kind, _)| kind == name)
            .map(|(_, item)| item.clone())
            .collect();
        if items.is_empty() {
            continue;
        }
        let scored = metrics(&items, temperature, ece_bins);
        by_type.insert(
            name.to_owned(),
            TypeMetrics {
                questions: items.len(),
                accuracy: scored.accuracy,
                nll: scored.nll,
            },
        );
    }
    by_type
}

/// `request` reduced to the questions `answers` labels.
fn labelled_request(request: &Request, answers: &BTreeMap<String, serde_json::Value>) -> Request {
    Request {
        state: request.state.clone(),
        model: None,
        questions: request
            .questions
            .iter()
            .filter(|(id, _)| answers.contains_key(*id))
            .map(|(id, question)| (id.clone(), question.clone()))
            .collect(),
    }
}

fn kind(question: &Question) -> &'static str {
    match question {
        Question::Choice { .. } => "choice",
        Question::Score { .. } => "score",
        Question::Noul { .. } => "noul",
    }
}
