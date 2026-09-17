//! `ster tune decide`: calibrated decision training on labelled decisions.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use ster::{decide::ExampleSet, lora, tune, ChatChoice, DecideTuneOptions, DeviceChoice, Precision, Runtime};

use super::super::super::ModelArgs;
use super::super::{note_precision, parse_adapter_layers, parse_targets};

/// `ster tune decide`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct DecideArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Labelled decisions: the same document `ster calibrate` reads,
    /// requests with the correct answer to some of their questions.
    #[arg(long)]
    examples: PathBuf,
    /// Output LoRA adapter safetensors; the identity sidecar is written beside it.
    #[arg(long)]
    output: PathBuf,
    /// Low-rank dimension shared by every adapter.
    #[arg(long, default_value_t = 8)]
    rank: usize,
    /// LoRA scaling numerator; each update is scaled by alpha over rank.
    #[arg(long, default_value_t = 16.0)]
    alpha: f64,
    /// Comma-separated projections to adapt: query, key, value, output,
    /// gate, up, or down.
    #[arg(long, default_value = "query,value")]
    targets: String,
    /// Comma-separated layers, half-open ranges such as 8..16, or all.
    #[arg(long, default_value = "all")]
    layers: String,
    /// Passes over the rows.
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
    /// Forwards folded into one optimizer step.
    #[arg(long, default_value_t = 8)]
    accumulation: usize,
    /// Steps over which the learning rate ramps up from zero.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    /// Renderings longer than this many tokens are skipped rather than
    /// truncated; a cut state is a different state.
    #[arg(long, default_value_t = 512)]
    max_sequence: usize,
    /// Option orders each question is trained in. 0 is every cyclic shift,
    /// so the letter is never predictive and the model has to read the
    /// state; 1 trains on one order and lets it learn a letter.
    #[arg(long, default_value_t = 0)]
    permutations: usize,
    /// auto renders every prompt through the model's own chat template
    /// when it publishes one, off sends raw text. Train in the format the
    /// decisions will be read in.
    #[arg(long, default_value = "auto")]
    chat_template: String,
    /// Rows folded into one forward pass.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
    /// Adapters and every optimizer moment stay in f32. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32")]
    precision: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub(in crate::cli::tune) fn decide(args: DecideArgs) -> Result<()> {
    let DecideArgs {
        model,
        examples,
        output,
        rank,
        alpha,
        targets,
        layers,
        epochs,
        learning_rate,
        accumulation,
        warmup_steps,
        max_sequence,
        permutations,
        chat_template,
        batch_size,
        precision,
        seed,
    } = args;
    // The examples are read before a weight is mapped, so a bad label is
    // refused in milliseconds.
    let example_set = ExampleSet::load(&examples)?;
    let device = DeviceChoice::parse(&model.device)?;
    let spec = lora::Spec {
        rank,
        alpha,
        targets: parse_targets(&targets)?,
        layers: parse_adapter_layers(&layers)?,
        seed,
    };
    let precision = Precision::parse(&precision)?;
    let (mut runtime, varmap) =
        Runtime::load_trainable_at(&model.model, model.revision.as_deref(), device, &spec, precision)?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&chat_template)?);
    let options = DecideTuneOptions {
        spec: spec.clone(),
        epochs,
        learning_rate,
        accumulation,
        batch: batch_size,
        warmup_steps,
        max_sequence,
        permutations,
        seed,
    };
    let report = tune::decide(&runtime, &varmap, &example_set, &options)?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
    artifact.save(&output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "path": output.display().to_string(),
            "report": report,
        }))?
    );
    Ok(())
}
