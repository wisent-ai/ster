//! `ster tune grpo`: policy optimization against a reward, with a sampled
//! group as its own baseline.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::json;
use ster::{
    ChatChoice, GenerationOptions, GrpoOptions, Precision, PromptSet, Reward, Runtime,
    UserSimulator, lora, tune,
};

use super::super::super::ModelArgs;
use super::super::{note_precision, parse_adapter_layers, parse_targets};

/// `ster tune grpo`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct GrpoArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file shaped as {"prompts": ["..."]}, the shape ster extract takes.
    #[arg(long)]
    prompts: PathBuf,
    /// Output LoRA adapter safetensors; the identity sidecar is written beside it.
    #[arg(long)]
    output: PathBuf,
    /// Where a completion's reward comes from: the keyword length, which
    /// counts sampled tokens and needs no judge; the path to a reward
    /// artifact written by ster tune reward; or an outside scorer,
    /// http(s)://host/path#/json/pointer, sent {"text": completion} (with
    /// STER_REWARD_BEARER as the bearer when set) and read at the pointer.
    #[arg(long, default_value = "length")]
    reward: String,
    /// Completions sampled per prompt. Their mean is the baseline, which is
    /// why two is the smallest group that means anything.
    #[arg(long, default_value_t = 4)]
    group: usize,
    /// Assistant turns per sampled conversation. Above one, --user-model
    /// writes each user turn in between, and a conversation's reward is
    /// the sum of its assistant turns' rewards.
    #[arg(long, default_value_t = 1)]
    turns: usize,
    /// The model that plays the user when --turns is above one: it
    /// answers the conversation with the roles swapped, at this run's
    /// sampling settings, and an empty reply ends the conversation. It
    /// must publish a chat template.
    #[arg(long)]
    user_model: Option<String>,
    /// Revision of --user-model.
    #[arg(long, requires = "user_model")]
    user_revision: Option<String>,
    /// Passes over the prompt set; each one re-samples from the current policy.
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    /// Weight on the KL penalty pulling the policy back to the frozen base.
    #[arg(long, default_value_t = 0.04)]
    beta: f64,
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
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f64,
    /// Prompt groups folded into one optimizer step. One group is already
    /// --group sequences, so a step per group is the natural unit.
    #[arg(long, default_value_t = 1)]
    accumulation: usize,
    /// Steps over which the learning rate ramps up from zero.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    /// Must exceed zero; argmax sampling would draw one identical
    /// completion per group and leave the baseline with nothing to compare.
    #[arg(long, default_value_t = 0.9)]
    temperature: f64,
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,
    /// Prompts whose prompt plus its longest completion exceed this many
    /// tokens are skipped rather than truncated.
    #[arg(long, default_value_t = 512)]
    max_sequence: usize,
    /// auto samples every completion from the prompt as the model's own
    /// chat template renders it, when it publishes one; off samples from
    /// raw text.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the frozen base weights are mapped at: f32, f16, or bf16.
    /// Adapters and every optimizer moment stay in f32. bf16 needs
    /// --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub(in crate::cli::tune) fn grpo(args: GrpoArgs) -> Result<()> {
    let GrpoArgs {
        model,
        prompts,
        output,
        reward,
        group,
        turns,
        user_model,
        user_revision,
        iterations,
        beta,
        rank,
        alpha,
        targets,
        layers,
        learning_rate,
        accumulation,
        warmup_steps,
        max_new_tokens,
        temperature,
        top_p,
        max_sequence,
        chat_template,
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
    // The reward source is resolved before the policy is loaded: a
    // reward artifact for the wrong checkpoint should be refused
    // before an operator waits out a policy load to hear it.
    let source = Reward::parse(&reward, &model.model, model.revision.as_deref(), device)?;
    let user = user_model
        .as_deref()
        .map(|name| UserSimulator::load(name, user_revision.as_deref(), device, precision))
        .transpose()?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &model.model,
        model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(chat_template);
    let prompt_set = PromptSet::load(&prompts)?;
    let options = GrpoOptions {
        spec: spec.clone(),
        group,
        turns,
        iterations,
        beta,
        learning_rate,
        accumulation,
        warmup_steps,
        max_sequence,
        generation: GenerationOptions {
            strength: 1.0,
            max_new_tokens,
            temperature,
            top_p: Some(top_p),
            seed,
        },
    };
    let report = tune::grpo(
        &runtime,
        &varmap,
        &prompt_set,
        &source,
        &reward,
        user.as_ref().zip(user_model.as_deref()),
        &options,
    )?;
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
