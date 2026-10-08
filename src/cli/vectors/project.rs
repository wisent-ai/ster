//! `ster vector project`: where a steering artifact moves the model's states
//! at one layer, projected onto the principal components of the unsteered
//! reads of a pair set.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
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
    let mut report = serde_json::to_value(workflow::project(&runtime, &pair_set, &artifact, &options)?)?;
    chat.annotate(&mut report)?;
    super::super::answer(&report)
}
