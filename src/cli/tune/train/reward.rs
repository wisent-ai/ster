//! `ster tune reward`: a scalar head that ranks the two sides of each pair.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use ster::{ChatChoice, PairSet, Precision, RewardHead, RewardOptions, Runtime, lora, tune};

use super::super::super::{ModelArgs, resolve_pairs};
use super::super::{note_precision, parse_adapter_layers, parse_targets};

/// `ster tune reward`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct RewardArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file with trait_name and contrastive pairs. The positive side
    /// is the response the head learns to score higher.
    /// Pair-set JSON. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    /// Output reward artifact safetensors, carrying the adapters and the
    /// head together; the identity sidecar is written beside it.
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
    /// than truncated; a cut response is not the response that was ranked.
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
    /// The adapters and the scalar head stay in f32. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    #[arg(long)]
    seed: u64,
}

pub(in crate::cli::tune) fn reward(args: RewardArgs) -> Result<()> {
    let RewardArgs {
        model,
        pairs,
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
    let pairs = resolve_pairs(pairs)?;
    let device = model.device;
    let spec = lora::Spec {
        rank,
        alpha,
        targets: parse_targets(&targets)?,
        layers: parse_adapter_layers(&layers)?,
        seed,
    };
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &model.model,
        model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(chat_template)?;
    // The head joins the same VarMap the adapters live in, so one
    // optimizer steps the pair and the artifact holds both. It is
    // registered at the parameter dtype, never the base dtype: a
    // scalar head is exactly the small trained weight that rounds away
    // in half precision.
    let head = RewardHead::fresh(
        &varmap,
        runtime.hidden_size(),
        runtime.device(),
        runtime.param_dtype(),
    )?;
    let pair_set = PairSet::load(&pairs)?;
    let options = RewardOptions {
        spec: spec.clone(),
        epochs,
        learning_rate,
        accumulation,
        batch: batch_size,
        warmup_steps,
        max_sequence,
        seed,
    };
    let report = tune::reward(&runtime, &varmap, &head, &pair_set, &options)?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.reward_artifact(&spec, head.weight(), report.clone())?;
    artifact.save(&output)?;
    crate::cli::answer(&json!({
        "path": output.display().to_string(),
        "report": report,
    }))?;
    Ok(())
}
