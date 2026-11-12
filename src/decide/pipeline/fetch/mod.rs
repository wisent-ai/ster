//! Labelled decisions from a Hugging Face classification dataset, read
//! through the datasets-server rows route: any set with a text column and a
//! label column becomes one choice question per row, with the label names
//! the dataset publishes as the options.

use std::collections::BTreeMap;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::{
    decide::{Example, ExampleSet, Question, Request},
    failures::{decide::fetch as causes, refuse},
    workflow,
};

mod page;

use page::{class_names, fields, read_page};

const ROWS_URL: &str = "https://datasets-server.huggingface.co/rows";

/// The most rows the route returns per page: the provider's own limit, "slices
/// of 100 rows maximum", https://huggingface.co/docs/dataset-viewer/en/quick_start.
const PAGE: usize = 100;

#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// `owner/name` on the Hub.
    pub dataset: String,
    pub config: String,
    pub split: String,
    /// The column holding the text a row is about.
    pub text_field: String,
    /// The column holding the class, as an index into the dataset's class
    /// names or as the name itself.
    pub label_field: String,
    /// The one question every example asks about its text.
    pub question_id: String,
    pub instructions: String,
    /// Rows to read, from `offset` into the split.
    pub count: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FetchReport {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub options: Vec<String>,
    pub requested: usize,
    pub rows: usize,
    pub examples: usize,
    pub skipped: usize,
    /// How many examples carry each label, so an ordered or lopsided slice
    /// is visible before anything is trained on it.
    pub label_counts: BTreeMap<String, usize>,
}

/// Reads `count` rows and turns each into an example with one labelled
/// choice question over the dataset's classes.
pub fn fetch(options: &FetchOptions) -> Result<(ExampleSet, FetchReport)> {
    if options.count == 0 {
        return Err(refuse(&causes::COUNT_NONE, &[]));
    }
    super::check_question("fetched", &options.question_id, &options.instructions)?;
    let agent = ureq::AgentBuilder::new().build();
    let mut names: Vec<String> = Vec::new();
    let mut examples = Vec::new();
    let mut rows = 0usize;
    let mut skipped = 0usize;
    let mut label_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut offset = options.offset;
    while rows < options.count {
        let length = PAGE.min(options.count - rows);
        workflow::progress(format!(
            "fetching rows {offset}..{} of {}",
            offset + length,
            options.dataset
        ));
        let page = read_page(&agent, options, offset, length)?;
        // The first page names the classes; `class_names` refuses fewer
        // than two, so an empty list means no page was read yet.
        if names.is_empty() {
            names = class_names(&page, options)?;
        }
        let Some(page_rows) = page.get("rows").and_then(Value::as_array) else {
            return Err(refuse(
                &causes::PAGE_ROWLESS,
                &[
                    ("dataset", &options.dataset),
                    ("offset", &offset.to_string()),
                    ("fields", &fields(&page)),
                ],
            ));
        };
        if page_rows.is_empty() {
            break;
        }
        for (position, entry) in page_rows.iter().enumerate() {
            rows += 1;
            let index = (offset + position).to_string();
            let row = &entry["row"];
            // A null cell is a row without text: skipped and counted.
            let text = match row.get(&options.text_field) {
                Some(Value::String(text)) => text.trim().to_owned(),
                Some(Value::Null) => String::new(),
                Some(other) => {
                    return Err(refuse(
                        &causes::TEXT_MISTYPED,
                        &[
                            ("dataset", &options.dataset),
                            ("index", &index),
                            ("text", &options.text_field),
                            ("found", &wisent_errors::trim_detail(&other.to_string(), 80)),
                        ],
                    ));
                }
                None => {
                    return Err(refuse(
                        &causes::TEXT_ABSENT,
                        &[
                            ("dataset", &options.dataset),
                            ("index", &index),
                            ("text", &options.text_field),
                            ("columns", &fields(row)),
                        ],
                    ));
                }
            };
            // A label the dataset's classes do not name (a negative, a
            // fraction, an index past them) leaves the row skipped, counted
            // in `none-kept`'s refusal when no row is kept.
            let label = match &row[&options.label_field] {
                Value::Number(number) => number.as_u64().and_then(|index| {
                    names
                        .iter()
                        .zip(0u64..)
                        .find(|(_, at)| *at == index)
                        .map(|(name, _)| name.clone())
                }),
                Value::String(name) => names.contains(name).then(|| name.clone()),
                _ => None,
            };
            let (Some(label), false) = (label, text.is_empty()) else {
                skipped += 1;
                continue;
            };
            *label_counts.entry(label.clone()).or_insert(0) += 1;
            examples.push(example(options, &names, text, label));
        }
        offset += page_rows.len();
        if page_rows.len() < length {
            break;
        }
    }
    if examples.is_empty() {
        return Err(refuse(
            &causes::NONE_KEPT,
            &[
                ("dataset", &options.dataset),
                ("first", &options.offset.to_string()),
                ("end", &offset.to_string()),
                ("rows", &rows.to_string()),
                ("skipped", &skipped.to_string()),
                ("classes", &names.join(", ")),
            ],
        ));
    }
    let set = ExampleSet { examples };
    set.validate()?;
    let report = FetchReport {
        dataset: options.dataset.clone(),
        config: options.config.clone(),
        split: options.split.clone(),
        options: names,
        requested: options.count,
        rows,
        examples: set.examples.len(),
        skipped,
        label_counts,
    };
    Ok((set, report))
}

fn example(options: &FetchOptions, classes: &[String], text: String, label: String) -> Example {
    let criteria: BTreeMap<String, Value> = classes
        .iter()
        .map(|name| (name.clone(), Value::Null))
        .collect();
    let question = Question::Choice {
        instructions: Value::String(options.instructions.clone()),
        criteria,
    };
    Example {
        request: Request {
            state: Value::String(text),
            model: None,
            questions: BTreeMap::from([(options.question_id.clone(), question)]),
        },
        answers: BTreeMap::from([(options.question_id.clone(), Value::String(label))]),
    }
}
