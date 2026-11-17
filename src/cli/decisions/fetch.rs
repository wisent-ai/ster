//! `ster decisions fetch`: labelled decisions from a Hugging Face dataset,
//! one choice question per row, its options either the class names of a
//! label column or each row's own choices.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use ster::{
    decide::{self, FetchColumns, FetchOptions},
    pairs::benchmark::{Benchmark, ChoiceFields},
};

/// The flags that locate a classification row's parts; each multiple-choice
/// flag refuses them.
const CLASS_FLAGS: &[&str] = &["text_field", "label_field"];

/// The flags that locate a multiple-choice row's parts; any of them stands in
/// for --text-field and --label-field.
const CHOICE_FLAGS: &[&str] = &["question", "choices", "answer", "answer_form", "labels"];

#[derive(Debug, clap::Args)]
pub(super) struct FetchArgs {
    /// Dataset id on the Hub, such as fancyzhx/ag_news.
    #[arg(long)]
    dataset: String,
    /// The dataset configuration on the Hub; none is assumed.
    #[arg(long)]
    config: String,
    /// The split read, such as train or test; none is assumed.
    #[arg(long)]
    split: String,
    /// The column holding the text each row is about. A classification set
    /// names it with --label-field; a multiple-choice set names --question,
    /// --choices, --answer and --answer-form instead.
    #[arg(long, required_unless_present_any = CHOICE_FLAGS)]
    text_field: Option<String>,
    /// The class-label column; its class names become the options.
    #[arg(long, required_unless_present_any = CHOICE_FLAGS)]
    label_field: Option<String>,
    /// Multiple-choice rows: the JSON pointer to a row's question, the state
    /// it is asked about, e.g. /question.
    #[arg(long, conflicts_with_all = CLASS_FLAGS)]
    question: Option<String>,
    /// Multiple-choice rows: the JSON pointer to a row's list of choices, its
    /// options, e.g. /choices/text.
    #[arg(long, conflicts_with_all = CLASS_FLAGS)]
    choices: Option<String>,
    /// Multiple-choice rows: the JSON pointer to a row's correct answer, e.g.
    /// /answerKey.
    #[arg(long, conflicts_with_all = CLASS_FLAGS)]
    answer: Option<String>,
    /// Multiple-choice rows: how the answer names the correct choice: index
    /// (from zero), label (one of --labels) or text (the choice itself).
    #[arg(long = "answer-form", conflicts_with_all = CLASS_FLAGS)]
    answer_form: Option<String>,
    /// Multiple-choice rows with --answer-form label: the JSON pointer to the
    /// row's labels, aligned with its choices, e.g. /choices/label.
    #[arg(long, conflicts_with_all = CLASS_FLAGS)]
    labels: Option<String>,
    /// The id the one question is asked under.
    #[arg(long)]
    question_id: String,
    /// The question asked about every text.
    #[arg(long)]
    instructions: String,
    /// Rows to read; every row of the split from --offset without it.
    #[arg(long)]
    count: Option<usize>,
    /// Rows to skip from the start of the split.
    #[arg(long, default_value_t)]
    offset: usize,
    /// Where the labelled decisions are written.
    #[arg(long)]
    output: PathBuf,
}

pub(super) fn fetch(args: FetchArgs) -> Result<()> {
    let columns = match (args.text_field, args.label_field) {
        (Some(text_field), Some(label_field)) => FetchColumns::Classes {
            text_field,
            label_field,
        },
        _ => {
            // The row flags are checked as `ster pairs import --benchmark
            // choices` checks them, before any row is read. Clap requires one
            // of them unless --text-field and --label-field are both given,
            // so `from_parts` answers the fields or its refusal.
            let fields = ChoiceFields::from_parts(
                Benchmark::Choices,
                args.question,
                args.choices,
                args.answer,
                args.answer_form,
                args.labels,
            )?;
            FetchColumns::Choices(fields.expect("clap requires a multiple-choice flag"))
        }
    };
    let options = FetchOptions {
        dataset: args.dataset,
        config: args.config,
        split: args.split,
        columns,
        question_id: args.question_id,
        instructions: args.instructions,
        count: args.count,
        offset: args.offset,
    };
    let (set, report) = decide::fetch(&options)?;
    super::write(&args.output, "the labelled decisions", &set)?;
    super::super::answer(&json!({"path": args.output.display().to_string(), "report": report}))?;
    Ok(())
}
