//! `ster vector project`: where a steering artifact moves the model's states
//! at one layer, projected onto the principal components of the unsteered
//! reads of a pair set; and `ster vector score`: a text's tokens, each
//! state at one layer projected onto the artifact's direction there.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use ster::{
    ChatChoice, PairSet, Precision, SteeringArtifact, tune,
    workflow::{self, ProjectOptions},
};

use super::super::{ModelArgs, resolve_pairs};

/// `ster vector project`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct ProjectArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Pair-set JSON. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    /// The steering artifact whose effect is projected.
    #[arg(long)]
    vector: PathBuf,
    /// The layer every side is read at.
    #[arg(long)]
    layer: usize,
    /// Strength the artifact is added at for the steered reads; Ster assumes none.
    #[arg(long)]
    strength: f64,
    /// How many principal components of the unsteered reads to project
    /// onto; Ster assumes none.
    #[arg(long)]
    components: NonZeroUsize,
    /// auto reads every side through the model's own chat template when it
    /// publishes one, off reads it as raw text; it should match the run that
    /// fitted the artifact.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

pub(in crate::cli) fn project(args: ProjectArgs) -> Result<()> {
    let ProjectArgs {
        model,
        pairs,
        vector,
        layer,
        strength,
        components,
        chat_template,
        precision,
    } = args;
    let pairs = resolve_pairs(pairs)?;
    // Both documents are read before a weight is mapped, so a wrong file is
    // refused before the checkpoint is paid for.
    let pair_set = PairSet::load(&pairs)?;
    let artifact = SteeringArtifact::load(&vector)?;
    let mut runtime = model.load_at(precision)?;
    let chat = runtime.set_chat_template(chat_template)?;
    tune::warn_on_provenance(&vector, "direction", &runtime);
    let options = ProjectOptions {
        layer,
        strength,
        components,
    };
    let mut report =
        serde_json::to_value(workflow::project(&runtime, &pair_set, &artifact, &options)?)?;
    chat.annotate(&mut report)?;
    super::super::answer(&report)
}

/// `ster vector score`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct ScoreArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// The steering artifact whose direction the tokens are scored along.
    #[arg(long)]
    vector: PathBuf,
    /// The layer every token's state is read at; the artifact must carry a vector there.
    #[arg(long)]
    layer: usize,
    /// The text to score. (`--text` is Ster's global switch for answers
    /// written for a person, so the scored text is the input.)
    #[arg(long, conflicts_with = "input_file")]
    input: Option<String>,
    /// A file holding the text to score.
    #[arg(long)]
    input_file: Option<PathBuf>,
    /// auto reads the text through the model's own chat template when it
    /// publishes one, off reads it as raw text; it should match the run that
    /// fitted the artifact.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

pub(in crate::cli) fn score(args: ScoreArgs) -> Result<()> {
    let ScoreArgs { model, vector, layer, input, input_file, chat_template, precision } = args;
    let text = match (input, input_file) {
        (Some(text), None) => text,
        (None, Some(file)) => {
            std::fs::read_to_string(&file).with_context(|| format!("--input-file {} cannot be read", file.display()))?
        }
        (None, None) => bail!("give the text to score with --input or --input-file"),
        (Some(_), Some(_)) => bail!("give the text once: --input or --input-file, not both"),
    };
    // The artifact is read before a weight is mapped, so a wrong file is
    // refused before the checkpoint is paid for.
    let artifact = SteeringArtifact::load(&vector)?;
    let mut runtime = model.load_at(precision)?;
    let chat = runtime.set_chat_template(chat_template)?;
    tune::warn_on_provenance(&vector, "direction", &runtime);
    let mut report = serde_json::to_value(workflow::score(&runtime, &artifact, &text, layer)?)?;
    chat.annotate(&mut report)?;
    super::super::answer(&report)
}
