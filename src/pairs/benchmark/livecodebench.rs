//! LiveCodeBench: each problem of problems.json paired with the good and
//! bad example code question_examples.json holds for its question_id.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::{Read, pair};
use crate::failures::{Rejection, file::json, pairs::export, pairs::rows as causes, refuse, shape};

/// What `--source` and `--examples` must hold, for the `ster.file` refusal
/// of one whose JSON does not read as that.
const PROBLEMS_FORM: &str = "LiveCodeBench's problems.json, one JSON list of problem objects";
const EXAMPLES_FORM: &str = "LiveCodeBench's question_examples.json, one JSON object whose questions list holds each question's examples";

/// The non-blank code of an example's `side` (`good_example` or
/// `bad_example`).
fn code(example: &Value, side: &str) -> Option<String> {
    example
        .pointer(&format!("/{side}/code"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .map(str::to_string)
}

/// What the top level of a JSON document is, for a refusal to show.
fn top(value: &Value) -> String {
    match value {
        Value::Object(fields) if fields.is_empty() => "an object with no keys".to_owned(),
        Value::Object(fields) => format!(
            "an object with the keys {}",
            fields.keys().cloned().collect::<Vec<_>>().join(", ")
        ),
        Value::Array(_) => "a list".to_owned(),
        Value::String(_) => "a string".to_owned(),
        Value::Number(_) => "a number".to_owned(),
        Value::Bool(_) => "a boolean".to_owned(),
        Value::Null => "null".to_owned(),
    }
}

/// The problems at `path` (read as `problems`) against the examples file
/// `examples_path` (read as `examples`).
pub(super) fn read(
    problems: &str,
    path: &str,
    examples: &str,
    examples_path: &Path,
) -> Result<Read> {
    let problems: Vec<Value> = serde_json::from_str(problems).map_err(|error| {
        json::refusal("the LiveCodeBench problems", PROBLEMS_FORM, Path::new(path), &error)
    })?;
    let shown = examples_path.display().to_string();
    let examples: Value = serde_json::from_str(examples).map_err(|error| {
        json::refusal("the LiveCodeBench examples", EXAMPLES_FORM, examples_path, &error)
    })?;
    let Some(examples) = examples.get("questions").and_then(Value::as_array) else {
        return Err(refuse(
            &export::EXAMPLES_NO_QUESTIONS,
            &[("path", &shown), ("top", &top(&examples))],
        ));
    };
    let mut found = Read::new(path, problems.len());
    for (position, problem) in problems.iter().enumerate() {
        let position = position + super::choices::FIRST_ROW;
        let Some(id) = problem
            .get("question_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        else {
            let values = vec![("found", shape(problem.get("question_id")))];
            let row = format!("problem {position}");
            found.skip(row.clone(), row, Rejection::new(&causes::PROBLEM_ID_MISSING, values));
            continue;
        };
        let place = format!("problem {position} ({id})");
        let Some(content) = problem
            .get("question_content")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|content| !content.is_empty())
        else {
            let values = vec![("found", shape(problem.get("question_content")))];
            found.skip(id, place, Rejection::new(&causes::PROBLEM_CONTENT_BLANK, values));
            continue;
        };
        let Some(example) = examples
            .iter()
            .find(|example| example.get("question_id").and_then(Value::as_str) == Some(id))
        else {
            let values = vec![
                ("id", id.to_owned()),
                ("examples", shown.clone()),
                ("count", examples.len().to_string()),
            ];
            found.skip(id, place, Rejection::new(&causes::EXAMPLE_MISSING, values));
            continue;
        };
        let (good, bad) = (code(example, "good_example"), code(example, "bad_example"));
        let (Some(good), Some(bad)) = (&good, &bad) else {
            let missing = match (&good, &bad) {
                (None, None) => "good_example nor bad_example",
                (None, Some(_)) => "good_example",
                _ => "bad_example",
            };
            let state = |side: &Option<String>| match side {
                Some(_) => "present".to_owned(),
                None => "absent or blank".to_owned(),
            };
            let values = vec![
                ("id", id.to_owned()),
                ("examples", shown.clone()),
                ("missing", missing.to_owned()),
                ("good", state(&good)),
                ("bad", state(&bad)),
            ];
            found.skip(id, place, Rejection::new(&causes::EXAMPLE_CODE_MISSING, values));
            continue;
        };
        found.pairs.push(pair(content, good, bad));
    }
    Ok(found)
}
