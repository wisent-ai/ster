//! Labelled decisions written by a hosted model through Brama.
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

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    brama::Gateway,
    decide::{request::text_of, Example, ExampleSet, Question, Request},
    workflow,
};

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
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let schema: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid decision schema {}", path.display()))?;
        if schema.domain.trim().is_empty() {
            bail!("a decision schema needs a domain");
        }
        if schema.questions.is_empty() {
            bail!("a decision schema needs at least one question");
        }
        Request { state: Value::String("schema".to_owned()), model: None, questions: schema.questions.clone() }
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
}

/// Sampling for a writer: warm enough that repeated asks differ, and a
/// budget that fits one state.
const TEMPERATURE: f64 = 0.9;
const MAX_TOKENS: usize = 160;

/// Writes `per_option` states for every option of every question through
/// `gateway`, each labelled with the option it was written for.
pub fn synthesize(gateway: &Gateway, schema: &Schema, options: &SynthesizeOptions) -> Result<(ExampleSet, SynthesizeReport)> {
    if options.per_option == 0 {
        bail!("synthesis needs at least one state per option");
    }
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
            let budget = options.per_option * options.retry_multiplier.max(1);
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
                let state = gateway.complete(&prompt, MAX_TOKENS, TEMPERATURE)?;
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
        bail!("the writer produced no usable state");
    }
    let set = ExampleSet { examples };
    set.validate()?;
    let report = SynthesizeReport {
        generator: format!("brama:{}", gateway.model()),
        domain: schema.domain.clone(),
        questions: schema.questions.len(),
        requested,
        attempts,
        kept: set.examples.len(),
        rejected_empty,
        rejected_duplicates,
    };
    Ok((set, report))
}

/// The options a question can be labelled with, as (name, text the writer
/// reads): option names for a choice, level texts for a score, yes and no
/// for a noul.
fn targets(question: &Question) -> Vec<(String, String)> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(name, description)| {
                let description = text_of(description);
                (name.clone(), if description.trim().is_empty() { name.clone() } else { format!("{name}: {description}") })
            })
            .collect(),
        Question::Score { criteria, .. } => {
            criteria.iter().enumerate().map(|(level, text)| (level.to_string(), text_of(text))).collect()
        }
        Question::Noul { criteria, .. } => {
            let criteria = criteria.clone().unwrap_or_default();
            vec![
                ("yes".to_owned(), text_of(&criteria.yes.unwrap_or(Value::String("yes".to_owned())))),
                ("no".to_owned(), text_of(&criteria.no.unwrap_or(Value::String("no".to_owned())))),
            ]
        }
    }
}

fn writer_prompt(domain: &str, question: &Question, target: &str) -> String {
    format!(
        "Write one realistic example of {domain}. Someone reading it and asked \"{}\" should clearly answer: {target}. Write only the example itself, two to four sentences, no preamble and no quotation marks.",
        text_of(question.instructions()).trim()
    )
}
