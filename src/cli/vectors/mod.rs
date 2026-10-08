//! `ster vector` — training a direction, choosing one, scoring one, reading
//! one, and comparing several — beside the commands that use a model the
//! same way: generating with it, exporting representations, parity, and
//! the first-use walkthrough.

use std::path::PathBuf;

use anyhow::{Context, Result};
use ster::{
    ChatChoice, PairSet, Precision, SteeringArtifact, TrainingMethod, tune,
    workflow::{self, parse_layers},
};

use super::onboarding;

mod compare;
mod generate;
mod project;
mod strength;

pub(super) use generate::{GenerateArgs, generate};

use super::{ModelArgs, resolve_pairs};

/// `ster vector`: every operation on a steering artifact.
#[derive(Debug, clap::Subcommand)]
pub(super) enum VectorCommand {
    /// Train steering vectors from positive and negative prompts.
    Train(TrainArgs),
    /// Select the best method and layer on a holdout of the pairs.
    Optimize(OptimizeArgs),
    /// Measure pair ordering for a steering artifact.
    Evaluate(EvaluateArgs),
    /// Summarize and validate a Ster steering artifact.
    Inspect(InspectArgs),
    /// Compare artifacts fitted for different traits on one model: cosine
    /// similarity per layer, what each holds that the others do not span,
    /// and the groups they fall into.
    Compare(compare::CompareArgs),
    /// Show where an artifact moves the model's states at one layer: every
    /// pair side read with and without it, projected onto the principal
    /// components of the unsteered reads.
    Project(project::ProjectArgs),
}

pub(super) fn run(command: VectorCommand) -> Result<()> {
    match command {
        VectorCommand::Train(args) => train(args),
        VectorCommand::Optimize(args) => optimize(args),
        VectorCommand::Evaluate(args) => evaluate(args),
        VectorCommand::Inspect(args) => inspect(args),
        VectorCommand::Compare(args) => compare::compare(args),
        VectorCommand::Project(args) => project::project(args),
    }
}

/// `ster vector train`
#[derive(Debug, clap::Args)]
pub(super) struct TrainArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Pair-set JSON. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    /// Output Ster steering artifact.
    #[arg(long)]
    output: PathBuf,
    /// Comma-separated layers, half-open ranges such as 8..16, or all.
    #[arg(long, default_value = "all")]
    layers: String,
    /// Direction training method: caa, pca, or logistic. Parsed by clap,
    /// so a typo is a usage error before a checkpoint is loaded.
    #[arg(long, default_value = "caa", value_parser = TrainingMethod::parse)]
    method: TrainingMethod,
    /// auto reads every pair through the model's own chat template when
    /// it publishes one, off reads it as raw text. A direction is fitted
    /// in whatever space the pairs were read in and added in whatever
    /// space generation runs in, so a direction fitted off and applied
    /// auto is measured in one space and steers another.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. A
    /// direction is fitted in whatever space the prompts were read in, so
    /// two artifacts trained at different precisions are not
    /// interchangeable. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

/// `ster vector optimize`
#[derive(Debug, clap::Args)]
pub(super) struct OptimizeArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Pair-set JSON. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "all")]
    layers: String,
    /// auto reads every pair through the model's own chat template when
    /// it publishes one, off reads it as raw text.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    /// Fraction of the pairs held out to rank candidates on, above zero and
    /// below one; Ster assumes none.
    #[arg(long)]
    holdout: f64,
}

/// `ster vector evaluate`
#[derive(Debug, clap::Args)]
pub(super) struct EvaluateArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Pair-set JSON. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    #[arg(long)]
    vector: PathBuf,
    /// auto reads every pair through the model's own chat template when
    /// it publishes one, off reads it as raw text. It should match the
    /// run that trained the artifact for the same reason --precision
    /// should.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. It should
    /// match the run that trained the artifact, or the score measures the
    /// direction in a space it was not fitted in.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    #[command(flatten)]
    strength: strength::StrengthArgs,
}

/// `ster extract`
#[derive(Debug, clap::Args)]
pub(super) struct ExtractArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file shaped as {"prompts": ["..."]}.
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "all")]
    layers: String,
    /// auto reads every prompt through the model's own chat template when
    /// it publishes one, off reads it as raw text. The exported
    /// activations are the states the model reached; this is what it was
    /// reading when it reached them.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. The
    /// exported activations are F32 either way; this is the width they
    /// were computed in. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

/// `ster parity`
#[derive(Debug, clap::Args)]
pub(super) struct ParityArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file shaped as {"records": [{"tokenIds": [...] or "tokenIdsFile":
    /// "ids.json", "hidden": "states.safetensors", "tensor": "hidden_states"}]}:
    /// token ids and the states after the final norm another implementation
    /// recorded for their first positions, [positions, hidden]; paths are
    /// relative to this file and "tensor" defaults to hidden_states.
    #[arg(long)]
    input: PathBuf,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. The states
    /// are compared in F32 either way; this is the width Ster computed them
    /// in. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

/// `ster vector inspect`
#[derive(Debug, clap::Args)]
pub(super) struct InspectArgs {
    #[arg(value_name = "ARTIFACT")]
    artifact: PathBuf,
}

/// `ster onboarding`
#[derive(Debug, clap::Args)]
pub(super) struct OnboardingArgs {
    /// Discard recorded progress and evidence, then show the walkthrough again.
    #[arg(long)]
    reset: bool,
    /// Existing canonical pair-set JSON to validate, persist, and make active.
    #[arg(long)]
    import_pairs: Option<PathBuf>,
    /// Stable workspace name; defaults to the source file name.
    #[arg(long, requires = "import_pairs")]
    name: Option<String>,
}

pub(super) fn train(args: TrainArgs) -> Result<()> {
    let TrainArgs {
        model,
        pairs,
        output,
        layers,
        method,
        chat_template,
        precision,
    } = args;
    let pairs = resolve_pairs(pairs)?;
    let mut runtime = model.load_at(precision)?;
    let chat = runtime.set_chat_template(chat_template)?;
    let pair_set = PairSet::load(&pairs)?;
    let layers = parse_layers(&layers, runtime.layer_count())?;
    let artifact = workflow::train(&runtime, &pair_set, &layers, method)?;
    artifact.save(&output)?;
    let mut summary = workflow::artifact_summary(&artifact);
    chat.annotate(&mut summary)?;
    super::answer(&summary)?;
    Ok(())
}
pub(super) fn optimize(args: OptimizeArgs) -> Result<()> {
    let OptimizeArgs {
        model,
        pairs,
        output,
        layers,
        chat_template,
        precision,
        holdout,
    } = args;
    let pairs = resolve_pairs(pairs)?;
    let mut runtime = model.load_at(precision)?;
    let chat = runtime.set_chat_template(chat_template)?;
    let pair_set = PairSet::load(&pairs)?;
    let layers = parse_layers(&layers, runtime.layer_count())?;
    let selection = workflow::optimize(&runtime, &pair_set, &layers, holdout)?;
    selection.artifact.save(&output)?;
    let mut summary = selection.summary();
    chat.annotate(&mut summary)?;
    super::answer(&summary)?;
    Ok(())
}
pub(super) fn evaluate(args: EvaluateArgs) -> Result<()> {
    let EvaluateArgs {
        model,
        pairs,
        vector,
        chat_template,
        precision,
        strength,
    } = args;
    let pairs = resolve_pairs(pairs)?;
    let mut runtime = model.load_at(precision)?;
    let chat = runtime.set_chat_template(chat_template)?;
    let pair_set = PairSet::load(&pairs)?;
    let artifact = SteeringArtifact::load(&vector)?;
    // The artifact now records the precision and the format it was
    // fitted in, so the advice `--precision` has always given can
    // finally be checked. Same helper the tune half uses.
    tune::warn_on_provenance(&vector, "direction", &runtime);
    let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
    let mut report = serde_json::to_value(report)?;
    if let Some(selection) = strength.measure(&runtime, &pair_set, &artifact)? {
        report
            .as_object_mut()
            .context("an evaluation report is a JSON object")?
            .insert("strength".to_owned(), serde_json::to_value(selection)?);
    }
    chat.annotate(&mut report)?;
    super::answer(&report)?;
    Ok(())
}
pub(super) fn extract(args: ExtractArgs) -> Result<()> {
    let ExtractArgs {
        model,
        input,
        output,
        layers,
        chat_template,
        precision,
    } = args;
    let mut runtime = model.load_at(precision)?;
    runtime.set_chat_template(chat_template)?;
    let layers = parse_layers(&layers, runtime.layer_count())?;
    workflow::extract(&runtime, &input, &output, &layers)?;
    println!("{}", output.display());
    Ok(())
}
pub(super) fn parity(args: ParityArgs) -> Result<()> {
    let ParityArgs {
        model,
        input,
        precision,
    } = args;
    let runtime = model.load_at(precision)?;
    super::answer(&workflow::parity(&runtime, &input)?)?;
    Ok(())
}
pub(super) fn inspect(args: InspectArgs) -> Result<()> {
    let InspectArgs { artifact } = args;
    let artifact = SteeringArtifact::load(&artifact)
        .with_context(|| format!("failed to inspect {}", artifact.display()))?;
    super::answer(&workflow::artifact_summary(&artifact))?;
    Ok(())
}
pub(super) fn onboarding(args: OnboardingArgs) -> Result<()> {
    let OnboardingArgs {
        reset,
        import_pairs,
        name,
    } = args;
    onboarding::run(reset, import_pairs.as_deref(), name.as_deref())?;
    Ok(())
}
