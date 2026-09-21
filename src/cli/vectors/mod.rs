//! The steering arms of `ster`: training a direction, choosing one, scoring
//! one, generating with it, exporting representations, reading an artifact,
//! and the first-use walkthrough beside them.

use std::path::PathBuf;

use anyhow::{Context, Result};
use ster::{
    tune,
    workflow::{self, parse_layers},
    ChatChoice, DeviceChoice, GenerationOptions, PairSet, Precision, Runtime, SteeringArtifact,
    TrainingMethod,
};

use super::onboarding;

use super::{resolve_pairs, ModelArgs};

/// `ster train`
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
        /// Direction training method: caa, pca, or logistic.
        #[arg(long, default_value = "caa")]
        method: String,
        /// auto reads every pair through the model's own chat template when
        /// it publishes one, off reads it as raw text. A direction is fitted
        /// in whatever space the pairs were read in and added in whatever
        /// space generation runs in, so a direction fitted off and applied
        /// auto is measured in one space and steers another.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. A
        /// direction is fitted in whatever space the prompts were read in, so
        /// two artifacts trained at different precisions are not
        /// interchangeable. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
}

/// `ster optimize`
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
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. bf16 needs
        /// --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
}

/// `ster evaluate`
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
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. It should
        /// match the run that trained the artifact, or the score measures the
        /// direction in a space it was not fitted in.
        #[arg(long, default_value = "f32")]
        precision: String,
}

/// `ster generate`
#[derive(Debug, clap::Args)]
pub(super) struct GenerateArgs {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        vector: Option<PathBuf>,
        /// Frozen LoRA adapter artifact to load the model with. It must have
        /// been trained for this exact model: Ster refuses a mismatch rather
        /// than steering the wrong residual stream.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// auto renders the prompt through the model's own chat template when
        /// it publishes one, off sends the prompt as raw text. An instruct
        /// checkpoint asked a bare question continues the text instead of
        /// answering it, which is what auto exists to prevent.
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. Half
        /// precision holds a checkpoint in half the memory; a steering vector
        /// is cast to it on the way in. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
        #[arg(long, default_value_t = 1.0)]
        strength: f64,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Zero selects deterministic argmax generation.
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long)]
        top_p: Option<f64>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
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
        #[arg(long, default_value = "auto")]
        chat_template: String,
        /// Dtype the base weights are mapped at: f32, f16, or bf16. The
        /// exported activations are F32 either way; this is the width they
        /// were computed in. bf16 needs --device metal.
        #[arg(long, default_value = "f32")]
        precision: String,
}

/// `ster inspect`
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
    let TrainArgs { model, pairs, output, layers, method, chat_template, precision } = args;
            let pairs = resolve_pairs(pairs)?;
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let pair_set = PairSet::load(&pairs)?;
            let layers = parse_layers(&layers, runtime.layer_count())?;
            let method = TrainingMethod::parse(&method)?;
            let artifact = workflow::train(&runtime, &pair_set, &layers, method)?;
            artifact.save(&output)?;
            let mut summary = workflow::artifact_summary(&artifact);
            chat.annotate(&mut summary)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
pub(super) fn optimize(args: OptimizeArgs) -> Result<()> {
    let OptimizeArgs { model, pairs, output, layers, chat_template, precision } = args;
            let pairs = resolve_pairs(pairs)?;
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let pair_set = PairSet::load(&pairs)?;
            let layers = parse_layers(&layers, runtime.layer_count())?;
            let selection = workflow::optimize(&runtime, &pair_set, &layers)?;
            selection.artifact.save(&output)?;
            let mut summary = selection.summary();
            chat.annotate(&mut summary)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
pub(super) fn evaluate(args: EvaluateArgs) -> Result<()> {
    let EvaluateArgs { model, pairs, vector, chat_template, precision } = args;
            let pairs = resolve_pairs(pairs)?;
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let pair_set = PairSet::load(&pairs)?;
            let artifact = SteeringArtifact::load(&vector)?;
            // The artifact now records the precision and the format it was
            // fitted in, so the advice `--precision` has always given can
            // finally be checked. Same helper the tune half uses.
            tune::warn_on_provenance(&vector, "direction", &runtime);
            let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
            let mut report = serde_json::to_value(report)?;
            chat.annotate(&mut report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
pub(super) fn generate(args: GenerateArgs) -> Result<()> {
    let GenerateArgs {
            model,
            prompt,
            vector,
            adapter,
            chat_template,
            precision,
            strength,
            max_new_tokens,
            temperature,
            top_p,
            seed,
    } = args;
            let precision = Precision::parse(&precision)?;
            // Both documents are read before a single weight is mapped, so
            // the two halves of the wrong-document refusal cost the same. An
            // adapter was already refused this early because it is attached
            // during the load; a steering vector was not, and handing one the
            // wrong file paid for a full checkpoint load before being told.
            // `Reward::parse` resolves its source ahead of the policy load for
            // this reason and says so.
            let artifact = vector.as_deref().map(SteeringArtifact::load).transpose()?;
            // An adapter rewrites the projections themselves, so it is
            // attached while the weights are mapped rather than applied per
            // token the way a steering vector is.
            let mut runtime = match adapter.as_deref() {
                Some(adapter) => Runtime::load_with_adapter_at(
                    &model.model,
                    model.revision.as_deref(),
                    DeviceChoice::parse(&model.device)?,
                    adapter,
                    precision,
                )?,
                None => model.load_at(precision)?,
            };
            runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            if let Some(vector) = vector.as_deref() {
                tune::warn_on_provenance(vector, "direction", &runtime);
            }
            let generated = runtime.generate(
                &prompt,
                artifact.as_ref(),
                GenerationOptions { strength, max_new_tokens, temperature, top_p, seed },
            )?;
            println!("{generated}");
    Ok(())
}
pub(super) fn extract(args: ExtractArgs) -> Result<()> {
    let ExtractArgs { model, input, output, layers, chat_template, precision } = args;
            let mut runtime = model.load_at(Precision::parse(&precision)?)?;
            runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
            let layers = parse_layers(&layers, runtime.layer_count())?;
            workflow::extract(&runtime, &input, &output, &layers)?;
            println!("{}", output.display());
    Ok(())
}
pub(super) fn inspect(args: InspectArgs) -> Result<()> {
    let InspectArgs { artifact } = args;
            let artifact = SteeringArtifact::load(&artifact)
                .with_context(|| format!("failed to inspect {}", artifact.display()))?;
            println!("{}", serde_json::to_string_pretty(&workflow::artifact_summary(&artifact))?);
    Ok(())
}
pub(super) fn onboarding(args: OnboardingArgs) -> Result<()> {
    let OnboardingArgs { reset, import_pairs, name } = args;
            onboarding::run(reset, import_pairs.as_deref(), name.as_deref())?;
    Ok(())
}
