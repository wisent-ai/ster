//! `ster tune dpo`: preference optimization over a contrastive pair set.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use ster::{ChatChoice, DpoLoss, DpoOptions, PairSet, Precision, Runtime, lora, tune};

use super::super::super::{ModelArgs, resolve_pairs};
use super::super::{note_precision, parse_adapter_layers, parse_targets};

/// `ster tune dpo`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct DpoArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file with trait_name and contrastive pairs. The positive side
    /// is the chosen response and the negative side the rejected one, so a
    /// steering pair set trains a preference without being rewritten.
    /// Pair-set JSON. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    /// Output LoRA adapter safetensors; the identity sidecar is written beside it.
    #[arg(long)]
    output: PathBuf,
    /// Low-rank dimension shared by every adapter; Ster assumes none.
    #[arg(long)]
    rank: usize,
    /// LoRA scaling numerator; each update is scaled by alpha over rank.
    #[arg(long)]
    alpha: f64,
    /// Comma-separated projections to adapt: query, key, value, output,
    /// gate, up, or down.
    #[arg(long, default_value = "query,value")]
    targets: String,
    /// Comma-separated layers, half-open ranges such as 8..16, or all.
    #[arg(long, default_value = "all")]
    layers: String,
    /// How hard the frozen reference pulls the policy back.
    #[arg(long)]
    beta: f64,
    /// Preference objective: dpo for the sigmoid loss, ipo for the squared
    /// error against 1/(2*beta) over length-normalized log-probabilities.
    #[arg(long, default_value = "dpo")]
    loss: String,
    /// Passes over the pair set.
    #[arg(long)]
    epochs: usize,
    #[arg(long)]
    learning_rate: f64,
    /// Pairs folded into one optimizer step.
    #[arg(long)]
    accumulation: usize,
    /// Steps over which the learning rate ramps up from zero.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    /// Pairs with a side longer than this many tokens are skipped rather
    /// than truncated; a cut response is not the response that was preferred.
    #[arg(long)]
    max_sequence: usize,
    /// auto encodes both sides of every pair as the assistant turn the
    /// model's own chat template renders, when it publishes one; off
    /// encodes raw text.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Pairs folded into one forward pass; a pair is two rows.
    #[arg(long)]
    batch_size: usize,
    /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
    /// Adapters and every optimizer moment stay in f32. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    #[arg(long)]
    seed: u64,
}

pub(in crate::cli::tune) fn dpo(args: DpoArgs) -> Result<()> {
    let DpoArgs {
        model,
        pairs,
        output,
        rank,
        alpha,
        targets,
        layers,
        beta,
        loss,
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
    let pairs = resolve_pairs(pairs)?;
    let device = model.device;
    let spec = lora::Spec {
        rank,
        alpha,
        targets: parse_targets(&targets)?,
        layers: parse_adapter_layers(&layers)?,
        seed,
    };
    // The reference the objective measures against is this same
    // runtime with the adapters skipped, so exactly one model is
    // loaded however many times each sequence is scored.
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &model.model,
        model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(chat_template);
    let pair_set = PairSet::load(&pairs)?;
    let options = DpoOptions {
        spec: spec.clone(),
        loss: DpoLoss::parse(&loss)?,
        beta,
        epochs,
        learning_rate,
        accumulation,
        batch: batch_size,
        warmup_steps,
        max_sequence,
        seed,
    };
    let report = tune::dpo(&runtime, &varmap, &pair_set, &options)?;
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
