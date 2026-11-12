//! `ster extract`: exporting the representations a model reaches for prompts.

use std::path::PathBuf;

use anyhow::Result;
use ster::{
    ChatChoice, Precision,
    workflow::{self, parse_layers},
};

use super::super::ModelArgs;

/// `ster extract`
#[derive(Debug, clap::Args)]
pub(in super::super) struct ExtractArgs {
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

pub(in super::super) fn extract(args: ExtractArgs) -> Result<()> {
    let ExtractArgs {
        model,
        input,
        output,
        layers,
        chat_template,
        precision,
    } = args;
    // The prompt set is read before the weights, so a set that cannot be
    // read is refused before the checkpoint is paid for.
    let prompts = workflow::PromptSet::load(&input)?;
    let mut runtime = model.load_at(precision)?;
    runtime.set_chat_template(chat_template)?;
    let layers = parse_layers(&layers, runtime.layer_count())?;
    workflow::extract(&runtime, &prompts, &output, &layers)?;
    println!("{}", output.display());
    Ok(())
}
