//! Labelled decisions from a Hugging Face classification dataset, read
//! through the datasets-server rows route: any set with a text column and a
//! label column becomes one choice question per row, with the label names
//! the dataset publishes as the options.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

use crate::{
    decide::{Example, ExampleSet, MAX_OPTIONS, Question, Request},
    workflow,
};

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
        bail!("fetch needs a row count above zero");
    }
    if options.question_id.trim().is_empty() || options.instructions.trim().is_empty() {
        bail!("a fetched question needs an id and instructions");
    }
    let agent = ureq::AgentBuilder::new().build();
    let mut names: Option<Vec<String>> = None;
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
        if names.is_none() {
            names = Some(class_names(&page, &options.label_field)?);
        }
        let classes = names.as_ref().expect("names resolved on the first page");
        let Some(page_rows) = page.get("rows").and_then(Value::as_array) else {
            bail!(
                "the datasets server's page of {} at offset {offset} carries no `rows` array; the rows route at {ROWS_URL} answers with one",
                options.dataset
            );
        };
        if page_rows.is_empty() {
            break;
        }
        for (position, entry) in page_rows.iter().enumerate() {
            rows += 1;
            let index = offset + position;
            let row = &entry["row"];
            // A null cell is a row without text: skipped and counted.
            let text = match row.get(&options.text_field) {
                Some(Value::String(text)) => text.trim().to_owned(),
                Some(Value::Null) => String::new(),
                Some(other) => bail!(
                    "row {} of {} holds {other} in text column '{}', not a string; pass a text column as --text-field",
                    index,
                    options.dataset,
                    options.text_field
                ),
                None => bail!(
                    "row {} of {} has no column '{}'; pass one of its columns as --text-field",
                    index,
                    options.dataset,
                    options.text_field
                ),
            };
            let label = match &row[&options.label_field] {
                Value::Number(number) => number
                    .as_u64()
                    .and_then(|index| classes.get(index as usize))
                    .cloned(),
                Value::String(name) => classes.contains(name).then(|| name.clone()),
                _ => None,
            };
            let (Some(label), false) = (label, text.is_empty()) else {
                skipped += 1;
                continue;
            };
            *label_counts.entry(label.clone()).or_insert(0) += 1;
            examples.push(example(options, classes, text, label));
        }
        offset += page_rows.len();
        if page_rows.len() < length {
            break;
        }
    }
    let Some(names) = names else {
        bail!(
            "the datasets server returned no page of {}",
            options.dataset
        );
    };
    if examples.is_empty() {
        bail!(
            "no row of {} carried both a text and a known label",
            options.dataset
        );
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

fn read_page(
    agent: &ureq::Agent,
    options: &FetchOptions,
    offset: usize,
    length: usize,
) -> Result<Value> {
    let request = agent
        .get(ROWS_URL)
        .query("dataset", &options.dataset)
        .query("config", &options.config)
        .query("split", &options.split)
        .query("offset", &offset.to_string())
        .query("length", &length.to_string());
    let response = stado_wait::blocking(
        stado_wait::Kind::Network,
        format!("rows {offset}..{} of {}", offset + length, options.dataset),
        ROWS_URL,
        || request.call(),
    );
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Status(status, response)) => {
            let body = response.into_string().with_context(|| {
                format!(
                    "the datasets server answered {status} for {} with a body that could not be read",
                    options.dataset
                )
            })?;
            // The server words its refusal in an `error` field; any other
            // body is shown as it was sent.
            let sentence = match serde_json::from_str::<Value>(&body) {
                Ok(Value::Object(fields)) => match fields.get("error") {
                    Some(Value::String(error)) => error.clone(),
                    _ => body,
                },
                _ => body,
            };
            bail!(
                "the datasets server answered {status} for {}: {sentence}",
                options.dataset
            );
        }
        Err(ureq::Error::Transport(transport)) => {
            bail!("failed to reach the datasets server: {}", transport.kind());
        }
    };
    let text = response
        .into_string()
        .context("failed to read the datasets server's page")?;
    serde_json::from_str(&text).context("the datasets server returned a page that is not JSON")
}

/// The class names of `label_field`, from the page's feature list.
fn class_names(page: &Value, label_field: &str) -> Result<Vec<String>> {
    let features = page["features"]
        .as_array()
        .context("the page carries no features")?;
    let feature = features
        .iter()
        .find(|feature| feature["name"].as_str() == Some(label_field))
        .with_context(|| format!("the dataset has no column named '{label_field}'"))?;
    let Some(listed) = feature["type"]["names"].as_array() else {
        bail!(
            "column '{label_field}' is the feature {}, not a class label with named classes; pass a ClassLabel column as --label-field",
            feature["type"]
        );
    };
    let mut names = Vec::with_capacity(listed.len());
    for name in listed {
        let Some(name) = name.as_str() else {
            bail!("column '{label_field}' names a class {name}, not a string");
        };
        names.push(name.to_owned());
    }
    if names.len() < 2 {
        bail!("column '{label_field}' is not a class label with at least two named classes");
    }
    if names.len() > MAX_OPTIONS {
        bail!(
            "column '{label_field}' has {} classes; Ster labels at most {MAX_OPTIONS}",
            names.len()
        );
    }
    Ok(names)
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
