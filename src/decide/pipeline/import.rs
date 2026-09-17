//! Labelled decisions from a JSONL file of `{"context", "options", "label"}`
//! rows — the shape jevlike and other option-scoring datasets publish — so a
//! set built elsewhere trains and benchmarks here without a converter.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::decide::{Example, ExampleSet, Question, Request, MAX_OPTIONS};

#[derive(Debug, Deserialize)]
struct JsonlRow {
    context: String,
    options: Vec<String>,
    label: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub source: String,
    pub rows: usize,
    pub examples: usize,
    /// Rows refused, each with its line number and the sentence.
    pub rejected: Vec<String>,
}

/// Reads `path` line by line into one choice question per row. `question_id`
/// names the question every example asks and `instructions` is what it asks;
/// the row's options become the choice's options, in the row's order, and its
/// label index becomes the answer.
pub fn import_jsonl(path: &Path, question_id: &str, instructions: &str) -> Result<(ExampleSet, ImportReport)> {
    if question_id.trim().is_empty() {
        bail!("an imported question needs an id");
    }
    if instructions.trim().is_empty() {
        bail!("an imported question needs instructions");
    }
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut examples = Vec::new();
    let mut rejected = Vec::new();
    let mut rows = 0usize;
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let line_number = number + 1;
        let row: JsonlRow = match serde_json::from_str(line) {
            Ok(row) => row,
            Err(error) => {
                rejected.push(format!("line {line_number}: not a context/options/label row: {error}"));
                continue;
            }
        };
        if let Err(sentence) = check(&row) {
            rejected.push(format!("line {line_number}: {sentence}"));
            continue;
        }
        let criteria: BTreeMap<String, Value> =
            row.options.iter().map(|option| (option.trim().to_owned(), Value::Null)).collect();
        let question = Question::Choice { instructions: Value::String(instructions.to_owned()), criteria };
        examples.push(Example {
            request: Request {
                state: Value::String(row.context),
                model: None,
                questions: BTreeMap::from([(question_id.to_owned(), question)]),
            },
            answers: BTreeMap::from([(question_id.to_owned(), Value::String(row.options[row.label].trim().to_owned()))]),
        });
    }
    if examples.is_empty() {
        bail!("{} holds no importable rows", path.display());
    }
    let set = ExampleSet { examples };
    set.validate()?;
    let report = ImportReport { source: path.display().to_string(), rows, examples: set.examples.len(), rejected };
    Ok((set, report))
}

fn check(row: &JsonlRow) -> Result<(), String> {
    if row.context.trim().is_empty() {
        return Err("the context is empty".to_owned());
    }
    if row.options.len() < 2 {
        return Err("a row needs at least two options".to_owned());
    }
    if row.options.len() > MAX_OPTIONS {
        return Err(format!("a row has {} options; Ster labels at most {MAX_OPTIONS}", row.options.len()));
    }
    if row.options.iter().any(|option| option.trim().is_empty()) {
        return Err("an option is empty".to_owned());
    }
    let mut seen = std::collections::BTreeSet::new();
    if row.options.iter().any(|option| !seen.insert(option.trim())) {
        return Err("two options carry the same text".to_owned());
    }
    if row.label >= row.options.len() {
        return Err(format!("the label {} names no option of {}", row.label, row.options.len()));
    }
    Ok(())
}
