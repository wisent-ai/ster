//! `ster pairs import`: a pair set from a published benchmark export.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use serde_json::json;
use ster::pairs::benchmark::{self, Benchmark, ChoiceFields, ImportOptions};

#[derive(Debug, Args)]
pub(super) struct ImportArgs {
    /// Which export SOURCE is: truthfulqa, dna, livecodebench, bipo, or
    /// choices (multiple-choice rows of any dataset, read through --question,
    /// --choices, --answer and --answer-form).
    #[arg(long)]
    benchmark: String,
    /// The export: TruthfulQA_en.csv, DnA_en.csv, LiveCodeBench problems.json,
    /// one of the BiPO paper's behaviour CSVs (question,matching,not_matching),
    /// or multiple-choice rows as JSON Lines or a JSON list.
    #[arg(long)]
    source: PathBuf,
    /// LiveCodeBench good/bad example code; defaults to question_examples.json
    /// beside SOURCE.
    #[arg(long)]
    examples: Option<PathBuf>,
    /// choices: the JSON pointer to a row's question, e.g. /question.
    #[arg(long)]
    question: Option<String>,
    /// choices: the JSON pointer to a row's list of choices, e.g. /choices/text.
    #[arg(long)]
    choices: Option<String>,
    /// choices: the JSON pointer to a row's correct answer, e.g. /answerKey.
    #[arg(long)]
    answer: Option<String>,
    /// choices: how the answer names the correct choice: index (from zero),
    /// label (one of --labels) or text (the choice itself).
    #[arg(long = "answer-form")]
    answer_form: Option<String>,
    /// choices with --answer-form label: the JSON pointer to the row's labels,
    /// aligned with its choices, e.g. /choices/label.
    #[arg(long)]
    labels: Option<String>,
    /// Keep this many pairs, drawn with --seed; every pair when omitted.
    #[arg(long)]
    count: Option<usize>,
    /// Seed for the drawn pairs and TruthfulQA's pick among incorrect answers; Ster assumes none.
    #[arg(long)]
    seed: u64,
    /// Trait name written on the set; defaults to the benchmark name.
    #[arg(long = "trait")]
    trait_name: Option<String>,
    /// Pair-set JSON the pairs are written to.
    #[arg(long)]
    output: PathBuf,
}

pub(super) fn run(args: ImportArgs) -> Result<()> {
    let options = ImportOptions {
        benchmark: Benchmark::parse(&args.benchmark)?,
        trait_name: args.trait_name.unwrap_or_else(|| args.benchmark.clone()),
        source: args.source,
        examples: args.examples,
        fields: ChoiceFields::from_parts(
            args.question,
            args.choices,
            args.answer,
            args.answer_form,
            args.labels,
        )?,
        count: args.count,
        seed: args.seed,
    };
    let (set, report) = benchmark::import(&options)?;
    set.save(&args.output)?;
    super::super::answer(&json!({
        "output": args.output.display().to_string(),
        "trait_name": set.trait_name,
        "report": report,
    }))?;
    Ok(())
}
