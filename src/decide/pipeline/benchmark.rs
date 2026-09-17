//! Measuring a model on a held-out labelled set: how often it is right, how
//! honest its probabilities are, whether it read the state, and how long a
//! decision takes. One document per run, so a base model and the adapter
//! trained on it are compared on the same rows with the same numbers.

use std::{collections::BTreeMap, time::Instant};

use anyhow::Result;
use serde::Serialize;

use crate::{
    decide::{
        calibration::{metrics, Metrics},
        control_state, question_logits, ExampleSet, Labelled, Options, Question, Request,
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
    /// Every labelled question at the temperature the run used.
    pub metrics: Metrics,
    /// The same labels with each question judged against another example's
    /// state; absent below two examples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control: Option<Metrics>,
    /// Per question type.
    pub by_type: BTreeMap<String, TypeMetrics>,
    pub latency: Latency,
    /// Tokens the model processed across every example, prefix and suffixes.
    pub input_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeMetrics {
    pub questions: usize,
    pub accuracy: f64,
    pub nll: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Latency {
    /// Wall time over every example, model already loaded.
    pub total_seconds: f64,
    pub mean_ms_per_example: f64,
    pub mean_ms_per_question: f64,
}

const MILLISECONDS: f64 = 1000.0;

/// Reads every example of `set` and scores its labels.
pub fn benchmark(runtime: &Runtime, set: &ExampleSet, options: &BenchmarkOptions) -> Result<Benchmark> {
    set.validate()?;
    let count = set.examples.len();
    let mut labelled: Vec<(String, Labelled)> = Vec::new();
    let mut shuffled: Vec<Labelled> = Vec::new();
    let mut input_tokens = 0usize;
    let mut timed = 0f64;
    for (index, example) in set.examples.iter().enumerate() {
        workflow::progress(format!("benchmarking example {} of {count}", index + 1));
        let request = labelled_request(&example.request, &example.answers);
        let clock = Instant::now();
        let (logits, usage) = question_logits(runtime, &request, options.read)?;
        timed += clock.elapsed().as_secs_f64();
        input_tokens += usage.input_tokens;
        for ((id, question), logits) in request.questions.iter().zip(logits) {
            let truth = question.truth_index(&example.answers[id]).expect("validated label");
            labelled.push((kind(question).to_owned(), Labelled { logits, truth }));
        }
        if count > 1 {
            for (id, question) in &request.questions {
                let control = Request {
                    state: control_state(set, index, id).clone(),
                    model: None,
                    questions: BTreeMap::from([(id.clone(), question.clone())]),
                };
                let (logits, _) = question_logits(runtime, &control, options.read)?;
                let truth = question.truth_index(&example.answers[id]).expect("validated label");
                shuffled.push(Labelled { logits: logits.into_iter().next().expect("one question"), truth });
            }
        }
    }
    let temperature = options.read.temperature;
    let all: Vec<Labelled> = labelled.iter().map(|(_, item)| item.clone()).collect();
    let mut by_type = BTreeMap::new();
    for name in ["choice", "score", "noul"] {
        let items: Vec<Labelled> =
            labelled.iter().filter(|(kind, _)| kind == name).map(|(_, item)| item.clone()).collect();
        if items.is_empty() {
            continue;
        }
        let scored = metrics(&items, temperature);
        by_type.insert(name.to_owned(), TypeMetrics { questions: items.len(), accuracy: scored.accuracy, nll: scored.nll });
    }
    let overall = metrics(&all, temperature);
    let control = (count > 1).then(|| metrics(&shuffled, temperature));
    workflow::progress(format!(
        "{} labelled questions: accuracy {:.4}, nll {:.4}, ece {:.4}{}; {:.0} ms per example",
        all.len(),
        overall.accuracy,
        overall.nll,
        overall.ece,
        control.map_or(String::new(), |control| format!(", {:.4} on shuffled states", control.accuracy)),
        timed * MILLISECONDS / count as f64
    ));
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
        latency: Latency {
            total_seconds: timed,
            mean_ms_per_example: timed * MILLISECONDS / count as f64,
            mean_ms_per_question: timed * MILLISECONDS / all.len().max(1) as f64,
        },
        input_tokens,
    })
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
