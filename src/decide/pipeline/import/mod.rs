//! Labelled decisions from a JSONL file of `{"context", "options", "label"}`
//! rows — the shape jevlike and other option-scoring datasets publish — so a
//! set built elsewhere trains and benchmarks here without a converter. Each
//! line is read by [`row`], and a line it refuses is reported under its
//! declared cause (`ster.decisions.import.rows`).

mod row;

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::decide::{Example, ExampleSet, Question, Request};
use crate::failures::{Refused, Rejections, decide::pipeline as causes, file, refuse};

/// One refused row: its line, the refusal as a reader meets it, and its
/// declared cause (`functionality`, `id`, `docs`, `code`, `exit`, `usage`,
/// `retryable`, `remedy`).
#[derive(Debug, Clone, Serialize)]
pub struct Rejected {
    pub line: usize,
    #[serde(flatten)]
    pub refused: Refused,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub source: String,
    pub rows: usize,
    pub examples: usize,
    /// Rows refused, each with its line and its declared cause.
    pub rejected: Vec<Rejected>,
}

/// Reads `path` line by line into one choice question per row. `question_id`
/// names the question every example asks and `instructions` is what it asks;
/// the row's options become the choice's options, in the row's order, and its
/// label index becomes the answer.
pub fn import_jsonl(
    path: &Path,
    question_id: &str,
    instructions: &str,
) -> Result<(ExampleSet, ImportReport)> {
    super::check_question("imported", question_id, instructions)?;
    let text = fs::read_to_string(path)
        .map_err(|error| file::read::refusal("the JSONL export", path, &error))?;
    let mut examples = Vec::new();
    let mut rejected = Vec::new();
    let mut rejections = Rejections::default();
    let mut rows = 0usize;
    let shown = path.display().to_string();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let line_number = number + 1;
        let row = match row::read(line) {
            Ok(row) => row,
            Err(rejection) => {
                let rejection = rejection.with("line", line_number.to_string()).with("path", shown.as_str());
                rejected.push(Rejected {
                    line: line_number,
                    refused: rejections.record(&rejection),
                });
                continue;
            }
        };
        let criteria: BTreeMap<String, Value> = row
            .options
            .iter()
            .map(|option| (option.trim().to_owned(), Value::Null))
            .collect();
        let question = Question::Choice {
            instructions: Value::String(instructions.to_owned()),
            criteria,
        };
        examples.push(Example {
            request: Request {
                state: Value::String(row.context),
                model: None,
                questions: BTreeMap::from([(question_id.to_owned(), question)]),
            },
            answers: BTreeMap::from([(
                question_id.to_owned(),
                Value::String(row.options[row.label].clone()),
            )]),
        });
    }
    if examples.is_empty() {
        // The first refused row's own cause, so its fix is the row's, with
        // how every row was refused after it.
        let kept_none = "every row was refused, so no labelled decision could be written";
        return Err(match rejections.refusal(kept_none, rows) {
            Some(refusal) => refusal,
            None => refuse(
                &causes::ROWS_NONE,
                &[("path", &shown), ("state", &file::state(path))],
            ),
        });
    }
    let set = ExampleSet { examples };
    set.validate()?;
    let report = ImportReport {
        source: shown,
        rows,
        examples: set.examples.len(),
        rejected,
    };
    Ok((set, report))
}
