//! `ster decisions`: the pipeline around a decision model — getting labelled
//! decisions, splitting them, and benchmarking a model on them. Training is
//! `ster tune decide`, reading is `ster decide`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Subcommand;
use serde_json::json;
use ster::{
    ChatChoice, DecideOptions, Precision, RAW_TEMPERATURE, Runtime,
    decide::{self, Calibration, ExampleSet, Schema, Steering, SynthesizeOptions},
    inference::Inference,
};

use super::{ModelArgs, vectors::SteeringArgs};

mod fetch;

#[derive(Debug, Subcommand)]
pub(super) enum DecisionsCommand {
    /// Read a Hugging Face classification or multiple-choice dataset into labelled decisions.
    Fetch(fetch::FetchArgs),
    /// Read a JSONL file of {"context", "options", "label"} rows into labelled decisions.
    Import(ImportArgs),
    /// Write labelled decisions with a hosted model through Stado's inference alternatives, one label per state by construction.
    Synthesize(SynthesizeArgs),
    /// Split labelled decisions into a training set and a held-out set.
    Split(SplitArgs),
    /// Measure a model, with or without an adapter or steering artifacts, on held-out labelled decisions.
    Benchmark(BenchmarkArgs),
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
    /// The route the writer asks Stado's inference alternatives for: a Brama alias, a provider/model route or a selector.
    #[arg(long)]
    generator_model: String,
    /// States written per option of each question; Ster assumes none.
    #[arg(long)]
    per_option: usize,
    /// Attempts allowed per state before an option is given up on; Ster assumes none.
    #[arg(long)]
    retry_multiplier: usize,
    /// Sampling temperature of the writer; above zero, or every attempt replays one state. Without it the answering alternative's own applies.
    #[arg(long)]
    temperature: Option<f64>,
    /// The writer's token budget for one state; without it the answering alternative's own applies.
    #[arg(long)]
    max_tokens: Option<usize>,
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
    #[arg(long, default_value_t)]
    permutations: usize,
    /// Equal-width confidence bins the expected calibration error is
    /// measured over; Ster assumes none, and the document records it.
    #[arg(long)]
    ece_bins: usize,
    /// Where the benchmark document is written.
    #[arg(long)]
    output: PathBuf,
    /// Read every question a second time with these steering artifacts
    /// added, and report both reads and how many questions they changed.
    #[command(flatten)]
    steering: SteeringArgs,
}

pub(super) fn run(command: DecisionsCommand) -> Result<()> {
    match command {
        DecisionsCommand::Fetch(args) => fetch::fetch(args),
        DecisionsCommand::Import(args) => import(args),
        DecisionsCommand::Synthesize(args) => synthesize(args),
        DecisionsCommand::Split(args) => split(args),
        DecisionsCommand::Benchmark(args) => benchmark(args),
    }
}

/// Writes `set` to `path`, the file holding `role`.
fn write(path: &Path, role: &str, set: &impl serde::Serialize) -> Result<()> {
    let file = |action: &str, left: &str, error: std::io::Error| {
        ster::failures::file::write::refusal(action, role, path, left, &error)
    };
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| file("creating the directory of", "nothing was written", error))?;
    }
    std::fs::write(path, ster::json::pretty(set)).map_err(|error| {
        file(
            "writing",
            "a file that was there may be emptied or cut short",
            error,
        )
    })
}

fn import(args: ImportArgs) -> Result<()> {
    let (set, report) = decide::import_jsonl(&args.input, &args.question_id, &args.instructions)?;
    write(&args.output, "the labelled decisions", &set)?;
    super::answer(&json!({"path": args.output.display().to_string(), "report": report}))?;
    Ok(())
}

fn synthesize(args: SynthesizeArgs) -> Result<()> {
    // The route and the settings are option checks: refused before the
    // schema is read.
    let writer = Inference::new(&args.generator_model)?;
    let options = SynthesizeOptions {
        per_option: args.per_option,
        retry_multiplier: args.retry_multiplier,
        temperature: args.temperature,
        max_tokens: args.max_tokens,
    };
    options.check()?;
    let schema = Schema::load(&args.schema)?;
    let (set, report) = decide::synthesize(&writer, &schema, &options)?;
    write(&args.output, "the labelled decisions", &set)?;
    super::answer(&json!({"path": args.output.display().to_string(), "report": report}))?;
    Ok(())
}

fn split(args: SplitArgs) -> Result<()> {
    decide::check_holdout(args.holdout)?;
    let set = ExampleSet::load(&args.examples)?;
    let (train, holdout) = decide::split(&set, args.holdout, args.seed);
    write(&args.train_output, "the training set", &train)?;
    write(&args.holdout_output, "the held-out set", &holdout)?;
    super::answer(&json!({
        "train": {"path": args.train_output.display().to_string(), "examples": train.examples.len()},
        "holdout": {"path": args.holdout_output.display().to_string(), "examples": holdout.examples.len()},
        "seed": args.seed,
    }))?;
    Ok(())
}

fn benchmark(args: BenchmarkArgs) -> Result<()> {
    // The bin count is a setting: it is refused before the examples are read.
    decide::check_bins(args.ece_bins)?;
    let set = ExampleSet::load(&args.examples)?;
    let calibration = args
        .calibration
        .as_deref()
        .map(|path| Calibration::load(path).map(|document| (path, document)))
        .transpose()?;
    // Every steering artifact is read before a weight is mapped too.
    let artifacts = args.steering.artifacts()?;
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
    runtime.set_chat_template(args.chat_template)?;
    args.steering.warn_on_provenance(&runtime);
    let parts = args.steering.parts(&artifacts);
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
        steering: Steering::new(&runtime, &args.steering.vector, &parts)?,
    };
    let report = decide::benchmark(&runtime, &set, &options)?;
    // The report is printed first, so an output the disk refuses does not
    // lose a benchmark the model already ran.
    super::answer(&report)?;
    write(&args.output, "the benchmark report", &report)?;
    Ok(())
}
