//! `ster vector project`: where a steering artifact moves the model's states
//! at one layer, projected onto the principal components of the unsteered
//! reads of a pair set; and `ster vector score`: a text's tokens, each
//! state at one layer projected onto the artifact's direction there.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
use ster::{
    ChatChoice, PairSet, Precision, SteeringArtifact,
    failures::{file, refuse, workflow::project as causes},
    tune,
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
        ster::json::value(workflow::project(&runtime, &pair_set, &artifact, &options)?);
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
    #[arg(long)]
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
    let ScoreArgs {
        model,
        vector,
        layer,
        input,
        input_file,
        chat_template,
        precision,
    } = args;
    // The text is read and checked before the artifact and the weights, so
    // a missing or empty one is refused before the checkpoint is paid for.
    let (text, source) = match (input, input_file) {
        (Some(text), None) => (text, "--input".to_owned()),
        (None, Some(file)) => {
            let text = std::fs::read_to_string(&file).map_err(|error| {
                file::read::refusal_or(
                    &causes::INPUT_FILE_MISSING,
                    "the text to score",
                    &file,
                    &error,
                )
            })?;
            (text, format!("--input-file {}", file.display()))
        }
        (None, None) => {
            return Err(refuse(
                &causes::INPUT_ABSENT,
                &[
                    ("vector", &vector.display().to_string()),
                    ("layer", &layer.to_string()),
                ],
            ));
        }
        (Some(text), Some(file)) => {
            return Err(refuse(
                &causes::INPUT_TWICE,
                &[
                    ("characters", &text.chars().count().to_string()),
                    ("path", &file.display().to_string()),
                ],
            ));
        }
    };
    workflow::check_text(&text, &source)?;
    let artifact = SteeringArtifact::load(&vector)?;
    let mut runtime = model.load_at(precision)?;
    let chat = runtime.set_chat_template(chat_template)?;
    tune::warn_on_provenance(&vector, "direction", &runtime);
    let mut report = ster::json::value(workflow::score(&runtime, &artifact, &text, layer)?);
    chat.annotate(&mut report)?;
    super::super::answer(&report)
}
