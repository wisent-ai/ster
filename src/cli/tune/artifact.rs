//! What `ster tune` does with an adapter that already exists: fold it into
//! the base weights, score a checkpoint with it, or read it.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use serde_json::json;
use ster::{ChatChoice, EvaluateOptions, ExampleSet, Precision, Runtime, lora, tune};

use super::super::ModelArgs;
use super::note_precision;

/// `ster tune merge`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct MergeArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// The adapter to fold in. It must have been trained for this exact
    /// model, and must be a generation adapter rather than a reward model.
    #[arg(long)]
    adapter: PathBuf,
    /// Directory to write. It receives model.safetensors beside the
    /// source's own config.json and tokenizer.json, plus whichever of
    /// tokenizer_config.json and chat_template.jinja the source
    /// published, which together are what --model accepts. Those last
    /// two are where a chat template lives, so a source that published
    /// one merges to a checkpoint that still reports applied rather
    /// than absent.
    #[arg(long)]
    output: PathBuf,
}

/// `ster tune evaluate`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct EvaluateArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file shaped as {"examples": [{"prompt": "...", "completion": "..."}]}.
    #[arg(long)]
    examples: PathBuf,
    /// Frozen LoRA adapter to attach before scoring. Omit it to score the
    /// bare checkpoint, which is the run the adapter is compared against.
    #[arg(long)]
    adapter: Option<PathBuf>,
    /// Examples longer than this many tokens are skipped rather than
    /// truncated; a cut completion is not the completion being scored.
    /// Ster assumes none; it must match the run being compared against.
    #[arg(long)]
    max_sequence: usize,
    /// auto scores every example in the shape the model's own chat
    /// template renders, when it publishes one; off scores raw text. It
    /// must match the run that trained the adapter, or the score measures
    /// a format the adapter never saw.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Examples folded into one forward pass; Ster assumes none.
    #[arg(long)]
    batch_size: usize,
    /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
    /// A score is only comparable with another score taken at the same
    /// precision. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

/// `ster tune inspect`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct InspectArgs {
    #[arg(value_name = "ARTIFACT")]
    artifact: PathBuf,
}

pub(super) fn merge(args: MergeArgs) -> Result<()> {
    let MergeArgs {
        model,
        adapter,
        output,
    } = args;
    // No device and no runtime: merging rewrites tensors and never runs
    // the model, so it resolves the checkpoint's files without mapping
    // them.
    let report = tune::merge(&model.model, model.revision.as_deref(), &adapter, &output)?;
    crate::cli::answer(&json!({ "report": report }))?;
    Ok(())
}

pub(super) fn evaluate(args: EvaluateArgs) -> Result<()> {
    let EvaluateArgs {
        model,
        examples,
        adapter,
        max_sequence,
        chat_template,
        batch_size,
        precision,
    } = args;
    // The adapter is attached while the weights are mapped, exactly as
    // `generate --adapter` attaches one, so the score is the score of
    // the model an operator would actually run.
    let mut runtime = match adapter.as_deref() {
        Some(adapter) => Runtime::load_with_adapter_at(
            &model.model,
            model.revision.as_deref(),
            model.device,
            adapter,
            precision,
        )?,
        None => Runtime::load_at(
            &model.model,
            model.revision.as_deref(),
            model.device,
            precision,
        )?,
    };
    let chat = runtime.set_chat_template(chat_template)?;
    let example_set = ExampleSet::load(&examples)?;
    let report = tune::evaluate(
        &runtime,
        &example_set,
        adapter.as_deref(),
        &EvaluateOptions {
            max_sequence,
            batch: batch_size,
        },
    )?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    crate::cli::answer(&report)?;
    Ok(())
}

pub(super) fn inspect(args: InspectArgs) -> Result<()> {
    let InspectArgs { artifact } = args;
    // Inspection reads the adapter document alone: no model is
    // loaded, so the tensors land on the CPU whatever trained them.
    let loaded = lora::Artifact::load(&artifact, &Device::Cpu)
        .with_context(|| format!("failed to inspect {}", artifact.display()))?;
    crate::cli::answer(&tune::inspect(&loaded))?;
    Ok(())
}
