//! Labelled decisions from a Hugging Face dataset, read through the
//! datasets-server rows route, one choice question per row: any set with a
//! text column and a label column, with the label names the dataset publishes
//! as the options, or any multiple-choice set (ARC, MMLU, TruthfulQA MC1,
//! HellaSwag), with each row's own question as the state and its own choices
//! as the options.

use std::collections::BTreeMap;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::{
    decide::{Example, ExampleSet, Question, Request},
    failures::{Refused, Rejections, decide::fetch as causes, refuse},
    pairs::benchmark::ChoiceFields,
    workflow,
};

mod choices;
mod page;

use page::{class_names, fields, read_page};

const ROWS_URL: &str = "https://datasets-server.huggingface.co/rows";

/// The most rows the route returns per page: the provider's own limit, "slices
/// of 100 rows maximum", https://huggingface.co/docs/dataset-viewer/en/quick_start.
const PAGE: usize = 100;

/// Where a dataset's rows hold what a labelled decision is made of.
#[derive(Debug, Clone)]
pub enum FetchColumns {
    /// A text column, the state, and a class-label column, whose class names
    /// are the options of every row.
    Classes {
        text_field: String,
        label_field: String,
    },
    /// Multiple-choice rows: each row's question is the state and its own
    /// choices are the options, found where the JSON pointers say, as `ster
    /// pairs import --benchmark choices` finds them.
    Choices(ChoiceFields),
}

#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// `owner/name` on the Hub.
    pub dataset: String,
    pub config: String,
    pub split: String,
    /// Where each row holds its state, its options and its answer.
    pub columns: FetchColumns,
    /// The one question every example asks about its state.
    pub question_id: String,
    pub instructions: String,
    /// Rows to read, from `offset` into the split; every row to its end when none.
    pub count: Option<usize>,
    pub offset: usize,
}

/// One multiple-choice row that could not become a labelled decision: its
/// index in the split, the refusal as a reader meets it, and its declared
/// cause.
#[derive(Debug, Clone, Serialize)]
pub struct RefusedRow {
    pub row: usize,
    #[serde(flatten)]
    pub refused: Refused,
}

#[derive(Debug, Clone, Serialize)]
pub struct FetchReport {
    pub dataset: String,
    pub config: String,
    pub split: String,
    /// The label column's class names; absent from multiple-choice rows,
    /// whose options are each row's own.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    /// The rows asked for; null when every row of the split from `offset` was.
    pub requested: Option<usize>,
    pub rows: usize,
    pub examples: usize,
    pub skipped: usize,
    /// How many examples carry each label (for multiple-choice rows, each
    /// position of the correct choice among the row's, from 0), so an ordered
    /// or lopsided slice is visible before anything is trained on it.
    pub label_counts: BTreeMap<String, usize>,
    /// Multiple-choice rows refused, each under its declared cause.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub refused: Vec<RefusedRow>,
}

/// What the rows read so far became.
#[derive(Default)]
struct Kept {
    examples: Vec<Example>,
    skipped: usize,
    label_counts: BTreeMap<String, usize>,
    refused: Vec<RefusedRow>,
    rejections: Rejections,
}

/// Reads `count` rows (every row from `offset` to the split's end when none)
/// and turns each into an example with one labelled choice question.
pub fn fetch(options: &FetchOptions) -> Result<(ExampleSet, FetchReport)> {
    if options.count == Some(0) {
        return Err(refuse(&causes::COUNT_NONE, &[]));
    }
    super::check_question("fetched", &options.question_id, &options.instructions)?;
    let agent = ureq::AgentBuilder::new().build();
    let mut names: Vec<String> = Vec::new();
    let mut kept = Kept::default();
    let mut rows: usize = Default::default();
    let mut offset = options.offset;
    // The label column whose class names are every row's options; none when
    // each row carries its own.
    let label = match &options.columns {
        FetchColumns::Classes { label_field, .. } => Some(label_field.as_str()),
        FetchColumns::Choices(_) => None,
    };
    while options.count.is_none_or(|count| rows < count) {
        // fallback: without --count every row of the split is read, a full
        // page at a time, as the flag's help states.
        let length = options.count.map_or(PAGE, |count| PAGE.min(count - rows));
        workflow::progress(format!(
            "fetching rows {offset}..{} of {}",
            offset + length,
            options.dataset
        ));
        let page = read_page(&agent, options, offset, length)?;
        // The first page names the classes; `class_names` refuses fewer
        // than two, so an empty list means no page was read yet.
        if let Some(label) = label.filter(|_| names.is_empty()) {
            names = class_names(&page, options, label)?;
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
            let index = offset + position;
            let row = &entry["row"];
            match &options.columns {
                FetchColumns::Classes {
                    text_field,
                    label_field,
                } => {
                    let columns = (text_field.as_str(), label_field.as_str());
                    class_row(options, &names, columns, index, row, &mut kept)?;
                }
                FetchColumns::Choices(located) => {
                    choices::read(options, located, index, row, &mut kept);
                }
            }
        }
        offset += page_rows.len();
        if page_rows.len() < length {
            break;
        }
    }
    if kept.examples.is_empty() {
        return Err(match &options.columns {
            FetchColumns::Classes { .. } => refuse(
                &causes::NONE_KEPT,
                &[
                    ("dataset", &options.dataset),
                    ("first", &options.offset.to_string()),
                    ("end", &offset.to_string()),
                    ("rows", &rows.to_string()),
                    ("skipped", &kept.skipped.to_string()),
                    ("classes", &names.join(", ")),
                ],
            ),
            FetchColumns::Choices(_) => choices::none_kept(options, kept.rejections, rows),
        });
    }
    let set = ExampleSet {
        examples: kept.examples,
    };
    set.validate()?;
    let report = FetchReport {
        dataset: options.dataset.clone(),
        config: options.config.clone(),
        split: options.split.clone(),
        options: names,
        requested: options.count,
        rows,
        examples: set.examples.len(),
        skipped: kept.skipped,
        label_counts: kept.label_counts,
        refused: kept.refused,
    };
    Ok((set, report))
}

/// Row `index` of a classification dataset, its text at column `text` and its
/// label at column `label`, kept as an example or skipped and counted.
fn class_row(
    options: &FetchOptions,
    names: &[String],
    (text, label): (&str, &str),
    index: usize,
    row: &Value,
    kept: &mut Kept,
) -> Result<()> {
    let shown = index.to_string();
    // A null cell is a row without text: skipped and counted.
    let state = match row.get(text) {
        Some(Value::String(state)) => state.trim().to_owned(),
        Some(Value::Null) => String::new(),
        Some(other) => {
            return Err(refuse(
                &causes::TEXT_MISTYPED,
                &[
                    ("dataset", &options.dataset),
                    ("index", &shown),
                    ("text", text),
                    ("found", &other.to_string().trim()),
                ],
            ));
        }
        None => {
            return Err(refuse(
                &causes::TEXT_ABSENT,
                &[
                    ("dataset", &options.dataset),
                    ("index", &shown),
                    ("text", text),
                    ("columns", &fields(row)),
                ],
            ));
        }
    };
    // A label the dataset's classes do not name (a negative, a fraction, an
    // index past them) leaves the row skipped, counted in `none-kept`'s
    // refusal when no row is kept.
    let named = match &row[label] {
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
    let (Some(named), false) = (named, state.is_empty()) else {
        kept.skipped += 1;
        return Ok(());
    };
    *kept.label_counts.entry(named.clone()).or_default() += 1;
    let criteria = names
        .iter()
        .map(|name| (name.clone(), Value::Null))
        .collect();
    kept.examples.push(example(options, state, criteria, named));
    Ok(())
}

/// The example asking `options`' one question about `state`, among the
/// options `criteria` names, labelled `answer`.
fn example(
    options: &FetchOptions,
    state: String,
    criteria: BTreeMap<String, Value>,
    answer: String,
) -> Example {
    let question = Question::Choice {
        instructions: Value::String(options.instructions.clone()),
        criteria,
    };
    Example {
        request: Request {
            state: Value::String(state),
            model: None,
            questions: BTreeMap::from([(options.question_id.clone(), question)]),
        },
        answers: BTreeMap::from([(options.question_id.clone(), Value::String(answer))]),
    }
}
