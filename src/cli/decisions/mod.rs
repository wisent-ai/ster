//! `ster decisions`: the pipeline around a decision model — getting labelled
//! decisions, splitting them, and benchmarking a model on them. Training is
//! `ster tune decide`, reading is `ster decide`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Subcommand;
use serde_json::json;
use ster::{
    ChatChoice, DecideOptions, Precision, RAW_TEMPERATURE, Runtime,
    brama::Gateway,
    decide::{self, Calibration, ExampleSet, FetchOptions, Schema, SynthesizeOptions},
};

use super::ModelArgs;

#[derive(Debug, Subcommand)]
pub(super) enum DecisionsCommand {
    /// Read a Hugging Face classification dataset into labelled decisions.
    Fetch(FetchArgs),
    /// Read a JSONL file of {"context", "options", "label"} rows into labelled decisions.
    Import(ImportArgs),
    /// Write labelled decisions with a hosted model through Brama, one label per state by construction.
    Synthesize(SynthesizeArgs),
    /// Split labelled decisions into a training set and a held-out set.
    Split(SplitArgs),
    /// Measure a model, with or without an adapter, on held-out labelled decisions.
    Benchmark(BenchmarkArgs),
}

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
    /// The column holding the text each row is about.
    #[arg(long)]
    text_field: String,
    /// The class-label column; its class names become the options.
    #[arg(long)]
    label_field: String,
    /// The id the one question is asked under.
    #[arg(long)]
    question_id: String,
    /// The question asked about every text.
    #[arg(long)]
    instructions: String,
    /// Rows to read.
    #[arg(long)]
    count: usize,
    /// Rows to skip from the start of the split.
    #[arg(long, default_value_t = 0)]
    offset: usize,
    /// Where the labelled decisions are written.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(super) struct ImportArgs {
    /// The JSONL file, one {"context", "options", "label"} row per line.
    #[arg(long)]
    input: PathBuf,
    /// The id the one question is asked under.
    #[arg(long)]
    question_id: String,
    /// The question asked about every context.
    #[arg(long)]
    instructions: String,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(super) struct SynthesizeArgs {
    /// The schema: a domain and the questions, with no state.
    #[arg(long)]
    schema: PathBuf,
    /// A Brama alias or route the writer runs on.
    #[arg(long)]
    generator_model: String,
    /// States written per option of each question; Ster assumes none.
    #[arg(long)]
    per_option: usize,
    /// Attempts allowed per state before an option is given up on; Ster assumes none.
    #[arg(long)]
    retry_multiplier: usize,
    /// Sampling temperature of the writer; above zero, or every attempt replays one state.
    #[arg(long)]
    temperature: f64,
    /// The writer's token budget for one state.
    #[arg(long)]
    max_tokens: usize,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(super) struct SplitArgs {
    #[arg(long)]
    examples: PathBuf,
    /// Fraction of examples held out; Ster assumes none.
    #[arg(long)]
    holdout: f64,
    /// The seed of the shuffle the held-out examples are drawn from.
    #[arg(long)]
    seed: u64,
    /// Where the training examples are written.
    #[arg(long)]
    train_output: PathBuf,
    /// Where the held-out examples are written.
    #[arg(long)]
    holdout_output: PathBuf,
}

#[derive(Debug, clap::Args)]
pub(super) struct BenchmarkArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Held-out labelled decisions.
    #[arg(long)]
    examples: PathBuf,
    /// A LoRA adapter trained for this exact model, such as one from `ster tune decide`.
    #[arg(long)]
    adapter: Option<PathBuf>,
    /// A calibration written by `ster calibrate` for this exact model.
    #[arg(long)]
    calibration: Option<PathBuf>,
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    /// Option orders each question is shown in; 0 is every cyclic shift.
    #[arg(long, default_value_t = 0)]
    permutations: usize,
    /// Equal-width confidence bins the expected calibration error is
    /// measured over; Ster assumes none, and the document records it.
    #[arg(long)]
    ece_bins: usize,
    /// Where the benchmark document is written.
    #[arg(long)]
    output: PathBuf,
}

pub(super) fn run(command: DecisionsCommand) -> Result<()> {
    match command {
        DecisionsCommand::Fetch(args) => fetch(args),
        DecisionsCommand::Import(args) => import(args),
        DecisionsCommand::Synthesize(args) => synthesize(args),
        DecisionsCommand::Split(args) => split(args),
        DecisionsCommand::Benchmark(args) => benchmark(args),
    }
}

fn write(path: &Path, set: &ExampleSet) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(set)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn fetch(args: FetchArgs) -> Result<()> {
    let options = FetchOptions {
        dataset: args.dataset,
        config: args.config,
        split: args.split,
        text_field: args.text_field,
        label_field: args.label_field,
        question_id: args.question_id,
        instructions: args.instructions,
        count: args.count,
        offset: args.offset,
    };
    let (set, report) = decide::fetch(&options)?;
    write(&args.output, &set)?;
    super::answer(&json!({"path": args.output.display().to_string(), "report": report}))?;
    Ok(())
}

fn import(args: ImportArgs) -> Result<()> {
    let (set, report) = decide::import_jsonl(&args.input, &args.question_id, &args.instructions)?;
    write(&args.output, &set)?;
    super::answer(&json!({"path": args.output.display().to_string(), "report": report}))?;
    Ok(())
}

fn synthesize(args: SynthesizeArgs) -> Result<()> {
    let schema = Schema::load(&args.schema)?;
    let gateway = Gateway::from_env(&args.generator_model)?;
    let options = SynthesizeOptions {
        per_option: args.per_option,
        retry_multiplier: args.retry_multiplier,
        temperature: args.temperature,
        max_tokens: args.max_tokens,
    };
    let (set, report) = decide::synthesize(&gateway, &schema, &options)?;
    write(&args.output, &set)?;
    super::answer(&json!({"path": args.output.display().to_string(), "report": report}))?;
    Ok(())
}

fn split(args: SplitArgs) -> Result<()> {
    let set = ExampleSet::load(&args.examples)?;
    if !(args.holdout > 0.0 && args.holdout < 1.0) {
        anyhow::bail!("the held-out fraction must be above zero and below one");
    }
    let (train, holdout) = decide::split(&set, args.holdout, args.seed);
    write(&args.train_output, &train)?;
    write(&args.holdout_output, &holdout)?;
    super::answer(&json!({
        "train": {"path": args.train_output.display().to_string(), "examples": train.examples.len()},
        "holdout": {"path": args.holdout_output.display().to_string(), "examples": holdout.examples.len()},
        "seed": args.seed,
    }))?;
    Ok(())
}

fn benchmark(args: BenchmarkArgs) -> Result<()> {
    let set = ExampleSet::load(&args.examples)?;
    let calibration = args
        .calibration
        .as_deref()
        .map(|path| Calibration::load(path).map(|document| (path, document)))
        .transpose()?;
    let precision = args.precision;
    let device = args.model.device;
    let mut runtime = match args.adapter.as_deref() {
        Some(adapter) => Runtime::load_with_adapter_at(
            &args.model.model,
            args.model.revision.as_deref(),
            device,
            adapter,
            precision,
        )?,
        None => Runtime::load_at(
            &args.model.model,
            args.model.revision.as_deref(),
            device,
            precision,
        )?,
    };
    let mut read = DecideOptions {
        permutations: args.permutations,
        temperature: RAW_TEMPERATURE,
        explain: false,
    };
    if let Some((path, document)) = &calibration {
        document.check_model(path, &runtime.model_id)?;
        read.temperature = document.temperature;
    }
    runtime.set_chat_template(args.chat_template);
    let options = decide::BenchmarkOptions {
        read,
        calibration: calibration
            .as_ref()
            .map(|(path, _)| path.to_string_lossy().into_owned()),
        adapter: args
            .adapter
            .as_deref()
            .map(|path| path.to_string_lossy().into_owned()),
        ece_bins: args.ece_bins,
    };
    let report = decide::benchmark(&runtime, &set, &options)?;
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&args.output, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    super::answer(&report)?;
    Ok(())
}
