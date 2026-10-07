//! `ster tune bipo`: one steering vector learned by bi-directional preference
//! optimization over a contrastive pair set, written as a steering artifact
//! `ster generate --vector` applies.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use ster::{BipoOptions, ChatChoice, DpoLoss, PairSet, Precision, Runtime, tune};

use super::super::super::{ModelArgs, resolve_pairs};
use super::super::note_precision;

/// `ster tune bipo`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct BipoArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file with trait_name and contrastive pairs: the positive side is
    /// what +v makes the model prefer, the negative side what -v makes it
    /// prefer. Omit it to use the active imported set.
    #[arg(long)]
    pairs: Option<PathBuf>,
    /// Output steering artifact JSON.
    #[arg(long)]
    output: PathBuf,
    /// The layer whose residual stream the vector is added to.
    #[arg(long)]
    layer: usize,
    /// The multiple of the vector added while training, in each direction.
    #[arg(long)]
    strength: f64,
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
    /// Forwards folded into one optimizer step.
    #[arg(long)]
    accumulation: usize,
    /// Steps over which the learning rate ramps up from zero.
    #[arg(long)]
    warmup_steps: usize,
    /// Pairs with a side longer than this many tokens are skipped rather
    /// than truncated; a cut response is not the response that was preferred.
    #[arg(long)]
    max_sequence: usize,
    /// auto encodes both sides of every pair as the assistant turn the
    /// model's own chat template renders, when it publishes one; off
    /// encodes raw text. Generate in the format the vector was learned in.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Pairs folded into one forward pass; a pair is two rows, and every
    /// pair of one forward shares that forward's direction.
    #[arg(long)]
    batch_size: usize,
    /// Dtype the frozen base weights are mapped at: f32, f16, or bf16. The
    /// vector and every optimizer moment stay in f32. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    #[arg(long)]
    seed: u64,
}

pub(in crate::cli::tune) fn bipo(args: BipoArgs) -> Result<()> {
    let BipoArgs {
        model,
        pairs,
        output,
        layer,
        strength,
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
    let mut runtime = Runtime::load_at(
        &model.model,
        model.revision.as_deref(),
        model.device,
        precision,
    )?;
    let chat = runtime.set_chat_template(chat_template)?;
    let pair_set = PairSet::load(&pairs)?;
    let options = BipoOptions {
        layer,
        strength,
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
    let (artifact, report) = tune::bipo(&runtime, &pair_set, &options)?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    artifact.save(&output)?;
    crate::cli::answer(&json!({
        "path": output.display().to_string(),
        "report": report,
    }))?;
    Ok(())
}
