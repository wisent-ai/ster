//! `ster pairs import`: a pair set from a published benchmark export.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use serde_json::json;
use ster::pairs::benchmark::{self, Benchmark, ImportOptions};

#[derive(Debug, Args)]
pub(super) struct ImportArgs {
    /// Which export SOURCE is: truthfulqa, dna or livecodebench.
    #[arg(long)]
    benchmark: String,
    /// The export: TruthfulQA_en.csv, DnA_en.csv or LiveCodeBench problems.json.
    #[arg(long)]
    source: PathBuf,
    /// LiveCodeBench good/bad example code; defaults to question_examples.json
    /// beside SOURCE.
    #[arg(long)]
    examples: Option<PathBuf>,
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
