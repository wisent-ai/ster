//! Labelled decisions written by a hosted model through Stado's inference
//! alternatives.
//!
//! The label is known by construction: for every option of every question
//! in the schema, the writer is asked for states that clearly belong to that
//! option, and each state is recorded with that option as its answer. Nothing
//! judges the written state afterwards, so what the writer produces is what
//! the set says; a poor writer makes a poor set, and the benchmark on a
//! held-out split is what says so.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    decide::{Example, ExampleSet, Question, Request, request::text_of},
    failures::{decide::pipeline as causes, file, refuse},
    inference::Inference,
    workflow,
};

/// What a decision schema is called in a refusal of its file.
const ROLE: &str = "the decision schema";

/// The questions a synthesized set asks: a request with no state.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Schema {
    /// What the states are: the domain the writer is told to write in, such
    /// as "a customer support message to an online shoe store".
    pub domain: String,
    pub questions: BTreeMap<String, Question>,
}

impl Schema {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|error| file::read::refusal(ROLE, path, &error))?;
        let schema: Self = serde_json::from_slice(&bytes)
            .map_err(|error| file::json::refusal(ROLE, causes::SCHEMA_FORM, path, &error))?;
        let shown = path.display().to_string();
        if schema.domain.trim().is_empty() {
            return Err(refuse(
                &causes::DOMAIN_BLANK,
                &[
                    ("path", &shown),
                    ("domain", &format!("{:?}", schema.domain)),
                ],
            ));
        }
        if schema.questions.is_empty() {
            return Err(refuse(&causes::QUESTIONS_NONE, &[("path", &shown)]));
        }
        Request {
            state: Value::String("schema".to_owned()),
            model: None,
            questions: schema.questions.clone(),
        }
        .validate()?;
        Ok(schema)
    }
}

#[derive(Debug, Clone)]
pub struct SynthesizeOptions {
    /// States per option of each question.
    pub per_option: usize,
    /// Extra attempts allowed per state before giving up on an option.
    pub retry_multiplier: usize,
    /// Sampling temperature of the writer, as the caller stated it.
    pub temperature: f64,
    /// The writer's token budget for one state, as the caller stated it.
    pub max_tokens: usize,
}

impl SynthesizeOptions {
    /// Refuses settings that write nothing; checked before the schema is
    /// read and before any model is asked.
    pub fn check(&self) -> Result<()> {
        if self.per_option == 0 {
            return Err(refuse(&causes::PER_OPTION_NONE, &[]));
        }
        if self.retry_multiplier == 0 {
            return Err(refuse(&causes::RETRIES_NONE, &[]));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesizeReport {
    pub generator: String,
    pub domain: String,
    pub questions: usize,
    pub requested: usize,
    pub attempts: usize,
    pub kept: usize,
    pub rejected_empty: usize,
    pub rejected_duplicates: usize,
    /// The sampling the states were written with, as the caller stated it.
    pub temperature: f64,
    pub max_tokens: usize,
}

/// Writes `per_option` states for every option of every question through
/// `writer`, each labelled with the option it was written for.
pub fn synthesize(
    writer: &Inference,
    schema: &Schema,
    options: &SynthesizeOptions,
) -> Result<(ExampleSet, SynthesizeReport)> {
    options.check()?;
    let mut examples = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut attempts = 0usize;
    let mut rejected_empty = 0usize;
    let mut rejected_duplicates = 0usize;
    let mut requested = 0usize;
    for (id, question) in &schema.questions {
        for (index, (name, text)) in targets(question).into_iter().enumerate() {
            requested += options.per_option;
            let mut kept = 0usize;
            let budget = options.per_option * options.retry_multiplier;
            let mut tried = 0usize;
            while kept < options.per_option && tried < budget {
                tried += 1;
                attempts += 1;
                workflow::progress(format!(
                    "writing state {} of {} for question '{id}' option {name} (attempt {tried})",
                    kept + 1,
                    options.per_option
                ));
                let prompt = writer_prompt(&schema.domain, question, &text);
                let state = writer.complete(&prompt, options.max_tokens, options.temperature)?;
                let state = state.trim().trim_matches('"').trim().to_owned();
                if state.is_empty() {
                    rejected_empty += 1;
                    continue;
                }
                if !seen.insert(state.to_lowercase()) {
                    rejected_duplicates += 1;
                    continue;
                }
                let answer = match question {
                    Question::Choice { .. } => Value::String(name.clone()),
                    Question::Score { .. } => Value::from(index),
                    Question::Noul { .. } => Value::Bool(index == 0),
                };
                examples.push(Example {
                    request: Request {
                        state: Value::String(state),
                        model: None,
                        questions: BTreeMap::from([(id.clone(), question.clone())]),
                    },
                    answers: BTreeMap::from([(id.clone(), answer)]),
                });
                kept += 1;
            }
        }
    }
    if examples.is_empty() {
        return Err(refuse(
            &causes::NONE_WRITTEN,
            &[
                ("domain", &schema.domain),
                ("attempts", &attempts.to_string()),
                ("empty", &rejected_empty.to_string()),
                ("repeated", &rejected_duplicates.to_string()),
                (
                    "generator",
                    &format!("{}:{}", writer.provider(), writer.model()),
                ),
            ],
        ));
    }
    let set = ExampleSet { examples };
    set.validate()?;
    let report = SynthesizeReport {
        generator: format!("{}:{}", writer.provider(), writer.model()),
        domain: schema.domain.clone(),
        questions: schema.questions.len(),
        requested,
        attempts,
        kept: set.examples.len(),
        rejected_empty,
        rejected_duplicates,
        temperature: options.temperature,
        max_tokens: options.max_tokens,
    };
    Ok((set, report))
}

/// The options a question can be labelled with, as (name, text the writer
/// reads): option names for a choice, level indices for a score, yes and no
/// for a noul, each with the option line [`Question::options`] shows the
/// model.
fn targets(question: &Question) -> Vec<(String, String)> {
    let names: Vec<String> = match question {
        Question::Choice { criteria, .. } => criteria.keys().cloned().collect(),
        Question::Score { criteria, .. } => {
            (0..criteria.len()).map(|level| level.to_string()).collect()
        }
        Question::Noul { .. } => vec!["yes".to_owned(), "no".to_owned()],
    };
    names.into_iter().zip(question.options()).collect()
}

fn writer_prompt(domain: &str, question: &Question, target: &str) -> String {
    format!(
        "Write one realistic example of {domain}. Someone reading it and asked \"{}\" should clearly answer: {target}. Write only the example itself, two to four sentences, no preamble and no quotation marks.",
        text_of(question.instructions()).trim()
    )
}
