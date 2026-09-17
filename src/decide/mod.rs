//! decide — typed decisions from a local model, read rather than generated.
//!
//! A language model asked a question writes an answer one token at a time,
//! and everything that makes generated text hard for software to depend on —
//! the parsing, the refusals, the invented option, the missing confidence —
//! comes from that. A decision does not need any of it. When the options are
//! fixed in advance, the whole answer is already in the distribution the
//! model holds at one position: show it the state, the question and the
//! options under letters, and read how much mass it puts on each letter.
//! That is one forward pass with no sampling, it cannot produce an option
//! that was not offered, and it comes with a probability for every option.
//!
//! Two things make the number honest rather than merely available. The
//! options are shown in every cyclic order and the probabilities averaged,
//! because a model prefers some letters regardless of what they label. And
//! the logits can be divided by a temperature fitted on labelled decisions
//! (`ster calibrate`), so that `0.9` means right nine times in ten.
//!
//! The request and answer shapes are TypeSafe's System One shapes, so a
//! workflow written against that API runs against a local checkpoint here.

use std::collections::BTreeMap;

use anyhow::{bail, Result};

use crate::{workflow, Runtime};

mod answer;
mod calibration;
mod pipeline;
mod prompt;
mod request;

pub use answer::{Answer, Explanation, Order, QuestionLogits, Response, Usage};
pub use calibration::{Calibration, Labelled, Metrics, RAW_TEMPERATURE};
pub use pipeline::{
    benchmark, fetch, import_jsonl, split, synthesize, Benchmark, BenchmarkOptions, FetchOptions, FetchReport,
    ImportReport, Latency, Schema, SynthesizeOptions, SynthesizeReport, TypeMetrics,
};
pub use request::{Example, ExampleSet, NoulCriteria, Question, Request, MAX_OPTIONS};

/// How a decision run reads the model.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Option orders per question: `0` for every cyclic shift.
    pub permutations: usize,
    /// What the answer-letter logits are divided by before the softmax.
    pub temperature: f64,
    /// Whether the response carries per-order detail for every question.
    pub explain: bool,
}

/// Answer every question in `request`.
pub fn decide(
    runtime: &Runtime,
    request: &Request,
    options: Options,
    calibration: Option<&str>,
) -> Result<Response> {
    request.validate()?;
    let (logits, usage) = question_logits(runtime, request, options)?;
    let explain = options.explain.then(|| {
        request
            .questions
            .iter()
            .zip(&logits)
            .map(|((id, question), logits)| (id.clone(), Explanation::new(question.options(), logits)))
            .collect()
    });
    let answers = request
        .questions
        .iter()
        .zip(logits)
        .map(|((id, question), logits)| {
            let probabilities = logits.probabilities(options.temperature);
            (id.clone(), Answer::from_probabilities(question, probabilities))
        })
        .collect();
    Ok(Response {
        model: runtime.model_id.clone(),
        revision: runtime.revision.clone(),
        chat_template: runtime.chat_status().label(),
        precision: runtime.precision().name(),
        temperature: options.temperature,
        calibration: calibration.map(str::to_owned),
        permutations: options.permutations,
        answers,
        explain,
        usage,
    })
}

/// Fit a temperature on `examples` and report how the probabilities matched
/// the labels before and after.
///
/// The report also carries a control: the same questions judged against a
/// wrong state — for each example, the nearest later example whose answer to
/// the same question differs, so the control state contradicts the label
/// wherever the set allows it. A model that reads the state scores its labels
/// far better on the real pairing than on that one; a model that answers
/// from the options and the letters alone scores the same on both, and no
/// temperature can make that honest. One example gives nothing to pair
/// against, so the control is absent below two.
pub fn calibrate(runtime: &Runtime, examples: &ExampleSet, options: Options) -> Result<Calibration> {
    examples.validate()?;
    let count = examples.examples.len();
    let mut labelled = Vec::new();
    let mut shuffled = Vec::new();
    for (index, example) in examples.examples.iter().enumerate() {
        workflow::progress(format!("reading example {} of {count}", index + 1));
        labelled.extend(labelled_logits(runtime, example, &example.request.state, options)?);
        if count > 1 {
            for (id, question) in &example.request.questions {
                let Some(answer) = example.answers.get(id) else { continue };
                let other = control_state(examples, index, id);
                let one = Example {
                    request: Request {
                        state: other.clone(),
                        model: None,
                        questions: BTreeMap::from([(id.clone(), question.clone())]),
                    },
                    answers: BTreeMap::from([(id.clone(), answer.clone())]),
                };
                shuffled.extend(labelled_logits(runtime, &one, other, options)?);
            }
        }
    }
    let temperature = calibration::fit_temperature(&labelled);
    let before = calibration::metrics(&labelled, RAW_TEMPERATURE);
    let after = calibration::metrics(&labelled, temperature);
    let control = (count > 1).then(|| calibration::metrics(&shuffled, temperature));
    workflow::progress(format!(
        "fitted temperature {temperature:.4} over {} labelled questions: nll {:.4} -> {:.4}, ece {:.4} -> {:.4}, accuracy {:.4}{}",
        labelled.len(),
        before.nll,
        after.nll,
        before.ece,
        after.ece,
        after.accuracy,
        control.map_or(String::new(), |control| format!(" against {:.4} on shuffled states", control.accuracy))
    ));
    Ok(Calibration {
        schema: calibration::SCHEMA.to_owned(),
        model: runtime.model_id.clone(),
        revision: runtime.revision.clone(),
        chat_template: runtime.chat_status().label().to_owned(),
        precision: runtime.precision().name().to_owned(),
        temperature,
        examples: count,
        questions: labelled.len(),
        before,
        after,
        control,
    })
}

/// The logits of every labelled question in `example`, judged against
/// `state` — its own, or another example's for the control.
fn labelled_logits(
    runtime: &Runtime,
    example: &Example,
    state: &serde_json::Value,
    options: Options,
) -> Result<Vec<Labelled>> {
    let request = Request {
        state: state.clone(),
        model: None,
        questions: example.request.questions.clone(),
    };
    let (logits, _) = question_logits(runtime, &request, options)?;
    Ok(request
        .questions
        .iter()
        .zip(logits)
        .filter_map(|((id, question), logits)| {
            let answer = example.answers.get(id)?;
            let truth = question.truth_index(answer).expect("validated label");
            Some(Labelled { logits, truth })
        })
        .collect())
}

/// The state the control judges example `index`'s question `id` against:
/// the nearest later example, cyclically, whose answer to `id` differs from
/// this one's, or the next example when none differs.
///
/// Chosen this way rather than "the next example" because sets arrive in
/// label runs — a classification dataset's first hundred rows may all share
/// one class — and a control paired with a same-label state measures nothing.
pub fn control_state<'a>(set: &'a ExampleSet, index: usize, id: &str) -> &'a serde_json::Value {
    let count = set.examples.len();
    let own = set.examples[index].answers.get(id);
    let other = (1..count)
        .map(|step| &set.examples[(index + step) % count])
        .find(|other| other.answers.get(id).is_some_and(|answer| Some(answer) != own))
        .unwrap_or(&set.examples[(index + 1) % count]);
    &other.request.state
}

/// One sequence the model reads: which question, which option order, and the
/// tokens.
pub struct Row {
    pub question: usize,
    pub order: Vec<usize>,
    pub ids: Vec<u32>,
}

/// Every rendering of every question in `request`, tokenized, plus the token
/// ids each answer letter position may be spelled with. This is the one
/// place a decision prompt is turned into tokens, for reading and for
/// training alike, so the two cannot drift apart.
pub fn render_rows(
    runtime: &Runtime,
    request: &Request,
    permutations: usize,
) -> Result<(Vec<Row>, BTreeMap<usize, Vec<u32>>)> {
    let mut rows = Vec::new();
    let mut labels: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (index, (id, question)) in request.questions.iter().enumerate() {
        let texts = question.options();
        for position in 0..texts.len() {
            if let std::collections::btree_map::Entry::Vacant(slot) = labels.entry(position) {
                slot.insert(runtime.label_tokens(prompt::LABELS[position])?);
            }
        }
        for order in prompt::orders(texts.len(), permutations) {
            let text = prompt::render(request, question, &texts, &order);
            let ids = runtime.encode_prompt(&text)?;
            if ids.len() > runtime.context_length() {
                bail!(
                    "question '{id}' renders to {} tokens, past the {} this model was built for",
                    ids.len(),
                    runtime.context_length()
                );
            }
            rows.push(Row { question: index, order, ids });
        }
    }
    Ok((rows, labels))
}

/// The answer-letter logits of every question in `request`, in canonical
/// option order, plus what it cost.
///
/// The rows all begin with the same text — the header and the state — so the
/// longest run of tokens they share goes through the model once, and each row
/// then costs only what comes after it: its question, its options, and the
/// answer prompt. The shared run is found on the tokens rather than the text,
/// because a tokenizer may merge across the point where the texts diverge
/// and a prefix cut by characters could then end mid-token.
fn question_logits(
    runtime: &Runtime,
    request: &Request,
    options: Options,
) -> Result<(Vec<QuestionLogits>, Usage)> {
    if !(options.temperature.is_finite() && options.temperature > 0.0) {
        bail!("temperature must be a positive number");
    }
    let (rows, labels) = render_rows(runtime, request, options.permutations)?;
    let shared = shared_prefix(&rows);
    let suffixes: Vec<&[u32]> = rows.iter().map(|row| &row.ids[shared..]).collect();
    let mut usage = Usage::for_passes(usize::from(shared > 0) + rows.len());
    usage.input_tokens = shared + suffixes.iter().map(|suffix| suffix.len()).sum::<usize>();
    workflow::progress(format!(
        "reading the state once ({shared} tokens), then {} question renderings ({} tokens between them)",
        rows.len(),
        usage.input_tokens - shared
    ));
    let distributions = runtime.next_token_logits_after(&rows[0].ids[..shared], &suffixes)?;
    let mut logits: Vec<QuestionLogits> =
        request.questions.iter().map(|_| QuestionLogits::default()).collect();
    for (row, vocabulary) in rows.iter().zip(distributions) {
        let mut canonical = vec![0f32; row.order.len()];
        for (position, &option) in row.order.iter().enumerate() {
            canonical[option] =
                answer::log_sum_exp(labels[&position].iter().map(|&id| vocabulary[id as usize]));
        }
        logits[row.question].rows.push(canonical);
        logits[row.question].orders.push(row.order.clone());
    }
    Ok((logits, usage))
}

/// How many leading tokens every row has in common, short of any row's whole
/// length: each row must keep at least one token of its own, because the
/// position read is the last one and it has to be the row's, not the
/// prefix's.
fn shared_prefix(rows: &[Row]) -> usize {
    let first = &rows[0].ids;
    let longest = rows
        .iter()
        .map(|row| row.ids.iter().zip(first).take_while(|(a, b)| a == b).count())
        .min()
        .unwrap_or(0);
    let shortest = rows.iter().map(|row| row.ids.len()).min().unwrap_or(0);
    longest.min(shortest.saturating_sub(1))
}
