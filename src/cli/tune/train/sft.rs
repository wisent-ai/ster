//! `ster tune sft`: supervised fine-tuning from prompt and completion pairs.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use ster::{ChatChoice, ExampleSet, Precision, Runtime, SftOptions, lora, tune};

use super::super::super::ModelArgs;
use super::super::{note_precision, parse_adapter_layers, parse_targets};

/// `ster tune sft`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct SftArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file shaped as {"examples": [{"prompt": "...", "completion": "..."}]}.
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
    /// gate, up, or down. The default query,value is the pair the LoRA
    /// papers adapt first, and it is the cheapest useful choice.
    #[arg(long, default_value = "query,value")]
    targets: String,
    /// Comma-separated layers, half-open ranges such as 8..16, or all.
    #[arg(long, default_value = "all")]
    layers: String,
    /// Passes over the example set.
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
    /// Examples folded into one optimizer step.
    #[arg(long, default_value_t = 8)]
    accumulation: usize,
    /// Steps over which the learning rate ramps up from zero.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    /// Examples longer than this many tokens are skipped rather than
    /// truncated; a cut completion would teach the model to stop early.
    #[arg(long, default_value_t = 512)]
    max_sequence: usize,
    /// auto encodes every prompt and completion through the model's own
    /// chat template when it publishes one, off encodes raw text. An
    /// instruct checkpoint trained on raw text learns a format it will
    /// never be prompted in.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Examples folded into one forward pass. One is the unbatched pass
    /// every run recorded so far took; --accumulation still counts
    /// forwards, so a step sees up to batch-size times accumulation
    /// examples and nothing changes at the default.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
    /// Adapters, any head, and every optimizer moment stay in f32
    /// whatever this says, because a low-rank update below the weight's
    /// own ulp rounds to nothing in half. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub(in crate::cli::tune) fn sft(args: SftArgs) -> Result<()> {
    let SftArgs {
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
        chat_template,
        batch_size,
        precision,
        seed,
    } = args;
    let device = model.device;
    let spec = lora::Spec {
        rank,
        alpha,
        targets: parse_targets(&targets)?,
        layers: parse_adapter_layers(&layers)?,
        seed,
    };
    // The adapters have to exist before the first forward pass, so
    // the runtime is built from the spec rather than patched after
    // loading; the returned VarMap owns every trainable tensor.
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &model.model,
        model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(chat_template);
    let example_set = ExampleSet::load(&examples)?;
    let options = SftOptions {
        spec: spec.clone(),
        epochs,
        learning_rate,
        accumulation,
        batch: batch_size,
        warmup_steps,
        max_sequence,
        seed,
    };
    let report = tune::sft(&runtime, &varmap, &example_set, &options)?;
    // The report is folded into the artifact so a trained adapter
    // always carries the run that produced it, and the encoding it was
    // produced in travels with it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
    artifact.save(&output)?;
    crate::cli::answer(&json!({
        "path": output.display().to_string(),
        "report": report,
    }))?;
    Ok(())
}
