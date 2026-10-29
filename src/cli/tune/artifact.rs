//! Existing adapter operations return their actual document; CLI and native
//! request frontends own rendering, not a second implementation.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use serde_json::json;
use ster::{ChatChoice, EvaluateOptions, ExampleSet, Precision, Runtime, lora, tune};

use super::super::ModelArgs;
use super::note_precision;

use crate::cli::input::{self, InputCheck};

/// `ster tune merge`
#[derive(Debug, clap::Args, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MergeArgs {
    #[command(flatten)]
    #[serde(flatten)]
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

// The declared export format is a native protocol identity, not a classifier.
#[derive(Debug, Clone, Copy, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum ExportFormat {
    Peft,
}

/// `ster tune export`
#[derive(Debug, clap::Args, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ExportArgs {
    #[command(flatten)]
    #[serde(flatten)]
    model: ModelArgs,
    /// The adapter to write. It must have been trained for this exact
    /// model, and must be a generation adapter rather than a reward model.
    #[arg(long)]
    adapter: PathBuf,
    /// The layout to write. peft is the one there is: adapter_config.json
    /// and adapter_model.safetensors named after the base checkpoint's own
    /// projections, beside the base's tokenizer.json.
    #[arg(long, value_enum)]
    format: ExportFormat,
    /// Directory to write.
    #[arg(long)]
    output: PathBuf,
}

/// `ster tune evaluate`
#[derive(Debug, clap::Args, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EvaluateArgs {
    #[command(flatten)]
    #[serde(flatten)]
    model: ModelArgs,
    /// JSON file shaped as {"examples": [{"prompt": "...", "completion": "..."}]};
    /// an example may add "system": "..." for the system turn it is served under.
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
    #[serde(default = "input::default_chat_template")]
    chat_template: ChatChoice,
    /// Examples folded into one forward pass; Ster assumes none.
    #[arg(long)]
    batch_size: usize,
    /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
    /// A score is only comparable with another score taken at the same
    /// precision. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    #[serde(default = "input::default_precision")]
    precision: Precision,
}

/// `ster tune inspect`
#[derive(Debug, clap::Args, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InspectArgs {
    #[arg(value_name = "ARTIFACT")]
    artifact: PathBuf,
}

impl InputCheck for MergeArgs {
    fn check(&self) -> std::result::Result<(), String> {
        self.model.check()?;
        input::required_path(&self.adapter, "tune/merge adapter")?;
        input::required_path(&self.output, "tune/merge output")
    }
}

impl InputCheck for ExportArgs {
    fn check(&self) -> std::result::Result<(), String> {
        self.model.check()?;
        input::required_path(&self.adapter, "tune/export adapter")?;
        input::required_path(&self.output, "tune/export output")
    }
}

impl InputCheck for EvaluateArgs {
    fn check(&self) -> std::result::Result<(), String> {
        self.model.check()?;
        input::required_path(&self.examples, "tune/evaluate examples")?;
        if let Some(adapter) = &self.adapter {
            input::required_path(adapter, "tune/evaluate adapter")?;
        }
        Ok(())
    }
}

impl InputCheck for InspectArgs {
    fn check(&self) -> std::result::Result<(), String> {
        input::required_path(&self.artifact, "tune/inspect artifact")
    }
}

pub(super) fn merge(args: MergeArgs) -> Result<serde_json::Value> {
    let MergeArgs {
        model,
        adapter,
        output,
    } = args;
    // No device and no runtime: merging rewrites tensors and never runs
    // the model, so it resolves the checkpoint's files without mapping
    // them.
    let report = tune::merge(&model.model, model.revision.as_deref(), &adapter, &output)?;
    Ok(json!({ "report": report }))
}

pub(super) fn export(args: ExportArgs) -> Result<serde_json::Value> {
    let ExportArgs {
        model,
        adapter,
        format,
        output,
    } = args;
    // Like merge, export renames tensors and never runs the model.
    let report = match format {
        ExportFormat::Peft => {
            tune::export_peft(&model.model, model.revision.as_deref(), &adapter, &output)?
        }
    };
    Ok(json!({ "report": report }))
}

pub(super) fn evaluate(args: EvaluateArgs) -> Result<serde_json::Value> {
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
    Ok(report)
}

pub(super) fn inspect(args: InspectArgs) -> Result<serde_json::Value> {
    let InspectArgs { artifact } = args;
    // Inspection reads the adapter document alone: no model is
    // loaded, so the tensors land on the CPU whatever trained them.
    let loaded = lora::Artifact::load(&artifact, &Device::Cpu)
        .with_context(|| format!("failed to inspect {}", artifact.display()))?;
    Ok(tune::inspect(&loaded))
}
