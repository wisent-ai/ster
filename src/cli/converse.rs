//! `ster converse`: plays conversations between a model (optionally with an
//! adapter) and a simulated user, and writes each as `{"messages": [...]}`.
//!
//! It is the rollout `ster tune grpo --turns` trains on, run with no gradient,
//! so the conversations an evaluation reads are produced exactly the way the
//! policy's training conversations are.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::json;
use ster::{ChatChoice, GenerationOptions, Precision, PromptSet, Runtime, UserSimulator, tune};

use super::ModelArgs;

#[derive(Debug, clap::Args)]
pub(super) struct ConverseArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// A LoRA adapter written by ster tune sft or grpo, attached to --model.
    #[arg(long)]
    adapter: Option<PathBuf>,
    /// JSON file shaped as {"prompts": ["..."]}; each prompt opens one conversation.
    #[arg(long)]
    prompts: PathBuf,
    /// The model that plays the user; it answers each conversation with the
    /// roles swapped, and an empty reply ends the conversation.
    #[arg(long)]
    user_model: String,
    /// Revision of --user-model.
    #[arg(long, requires = "user_model")]
    user_revision: Option<String>,
    /// Assistant turns per conversation, at most.
    #[arg(long)]
    turns: usize,
    /// Tokens sampled per turn, for both models.
    #[arg(long)]
    max_new_tokens: usize,
    /// A conversation plus the next turn may not exceed this many tokens.
    #[arg(long)]
    max_sequence: usize,
    #[arg(long)]
    temperature: f64,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long)]
    seed: u64,
    /// Dtype both models are mapped at: f32, f16, or bf16.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    /// JSONL file the conversations are written to.
    #[arg(long)]
    output: PathBuf,
}

pub(super) fn run(args: ConverseArgs) -> Result<()> {
    if args.turns < 2 {
        bail!(
            "a conversation with a simulated user needs --turns of at least 2; one turn is ster generate"
        );
    }
    let prompts = PromptSet::load(&args.prompts)?;
    let user = UserSimulator::load(
        &args.user_model,
        args.user_revision.as_deref(),
        args.model.device,
        args.precision,
    )?;
    let mut runtime = match &args.adapter {
        Some(adapter) => Runtime::load_with_adapter_at(
            &args.model.model,
            args.model.revision.as_deref(),
            args.model.device,
            adapter,
            args.precision,
        )?,
        None => args.model.load_at(args.precision)?,
    };
    runtime.set_chat_template(ChatChoice::Auto);
    let generation = GenerationOptions {
        strength: 1.0,
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
    };
    let file = File::create(&args.output)
        .with_context(|| format!("cannot create {}", args.output.display()))?;
    let mut output = BufWriter::new(file);
    let mut draw = 0u64;
    let mut turns_played = 0usize;
    for (index, prompt) in prompts.prompts.iter().enumerate() {
        let played = tune::rollout(
            &runtime,
            Some(&user),
            prompt,
            args.turns,
            generation,
            args.max_sequence,
            &mut draw,
        )
        .with_context(|| format!("prompt {index} could not be played"))?;
        turns_played += played.turns.len();
        let messages: Vec<_> = played
            .transcript
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect();
        writeln!(
            output,
            "{}",
            json!({"prompt_index": index, "messages": messages})
        )?;
    }
    output
        .flush()
        .with_context(|| format!("cannot write {}", args.output.display()))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "output": args.output.display().to_string(),
            "conversations": prompts.prompts.len(),
            "assistant_turns": turns_played,
            "user_model": args.user_model,
            "adapter": args.adapter.as_ref().map(|path| path.display().to_string()),
        }))?
    );
    Ok(())
}
