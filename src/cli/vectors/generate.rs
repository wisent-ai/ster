//! `ster generate`: text from a model, steered by any number of steering
//! artifacts each at its own strength, or through an adapter, for one prompt
//! or a prompt set.

use std::path::PathBuf;

use anyhow::{Context, Result};
use ster::{ChatChoice, GenerationOptions, Precision, Runtime, SteeringArtifact, tune, workflow};

use super::super::ModelArgs;

/// `ster generate`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct GenerateArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// The one prompt to answer; the answer is printed.
    #[arg(long, required_unless_present = "prompts", conflicts_with = "prompts")]
    prompt: Option<String>,
    /// A prompt set ({"prompts": ["..."]}) to answer one after the other
    /// with the model loaded once; the answers are written to --output.
    #[arg(long, requires = "output")]
    prompts: Option<PathBuf>,
    /// JSON file a --prompts run writes: one {"prompt", "model_output"} per
    /// prompt, in the set's order.
    #[arg(long, requires = "prompts")]
    output: Option<PathBuf>,
    /// File whose text is the system turn the prompt is answered under. It is
    /// rendered by the model's own chat template, so it needs one: with the
    /// template absent or `--chat-template off` it is refused.
    #[arg(long)]
    system: Option<PathBuf>,
    /// Steering artifact to add during generation; repeat it to steer
    /// several traits at once, each with the --strength in the same place.
    #[arg(long)]
    vector: Vec<PathBuf>,
    /// Frozen LoRA adapter artifact to load the model with. It must have
    /// been trained for this exact model: Ster refuses a mismatch rather
    /// than steering the wrong residual stream.
    #[arg(long)]
    adapter: Option<PathBuf>,
    /// auto renders the prompt through the model's own chat template when
    /// it publishes one, off sends the prompt as raw text. An instruct
    /// checkpoint asked a bare question continues the text instead of
    /// answering it, which is what auto exists to prevent.
    #[arg(long, default_value = "auto", value_parser = ChatChoice::parse)]
    chat_template: ChatChoice,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. Half
    /// precision holds a checkpoint in half the memory; a steering vector
    /// is cast to it on the way in. bf16 needs --device metal.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
    /// Scale on the --vector in the same place; one per --vector, which it
    /// belongs to, and Ster assumes none.
    #[arg(long, requires = "vector")]
    strength: Vec<f64>,
    #[arg(long)]
    max_new_tokens: usize,
    /// Zero selects deterministic argmax generation.
    #[arg(long)]
    temperature: f64,
    /// Nucleus mass; omitted samples from the whole distribution.
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long)]
    seed: u64,
}

pub(in crate::cli) fn generate(args: GenerateArgs) -> Result<()> {
    let GenerateArgs {
        model,
        prompt,
        prompts,
        output,
        system,
        vector,
        adapter,
        chat_template,
        precision,
        strength,
        max_new_tokens,
        temperature,
        top_p,
        seed,
    } = args;
    // Both documents are read before a single weight is mapped, so
    // the two halves of the wrong-document refusal cost the same. An
    // adapter was already refused this early because it is attached
    // during the load; a steering vector was not, and handing one the
    // wrong file paid for a full checkpoint load before being told.
    // `Reward::parse` resolves its source ahead of the policy load for
    // this reason and says so.
    let artifacts = vector
        .iter()
        .map(|path| SteeringArtifact::load(path))
        .collect::<Result<Vec<_>>>()?;
    if artifacts.len() != strength.len() {
        anyhow::bail!(
            "every --vector needs its own --strength: got {} vector(s) and {} strength(s); Ster assumes no steering scale",
            artifacts.len(),
            strength.len()
        );
    }
    let parts: Vec<(&SteeringArtifact, f64)> = artifacts.iter().zip(strength.iter().copied()).collect();
    // The system turn is read here for the same reason: an unreadable or
    // empty file is refused before the checkpoint is paid for.
    let system = match system.as_deref() {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read the system turn {}", path.display()))?;
            if text.trim().is_empty() {
                anyhow::bail!("the system turn {} is empty", path.display());
            }
            if prompt
                .as_deref()
                .is_some_and(|prompt| prompt.trim().is_empty())
            {
                anyhow::bail!("prompt must not be empty");
            }
            Some(text)
        }
        None => None,
    };
    // A prompt set is read before the weights too, so a malformed or empty
    // set is refused before the checkpoint is paid for.
    let prompt_set = prompts.as_deref().map(ster::PromptSet::load).transpose()?;
    // An adapter rewrites the projections themselves, so it is
    // attached while the weights are mapped rather than applied per
    // token the way a steering vector is.
    let mut runtime = match adapter.as_deref() {
        Some(adapter) => Runtime::load_with_adapter_at(
            &model.model,
            model.revision.as_deref(),
            model.device,
            adapter,
            precision,
        )?,
        None => model.load_at(precision)?,
    };
    runtime.set_chat_template(chat_template)?;
    for path in &vector {
        tune::warn_on_provenance(path, "direction", &runtime);
    }
    // Every part carries its own strength, so the options carry none.
    let options = GenerationOptions {
        strength: None,
        max_new_tokens,
        temperature,
        top_p,
        seed,
    };
    let answer = |prompt: &str| -> Result<String> {
        Ok(match system.as_deref() {
            Some(system) => {
                let context = runtime.encode_conversation(&[
                    ster::chat::Message {
                        role: "system",
                        content: system,
                    },
                    ster::chat::Message {
                        role: "user",
                        content: prompt,
                    },
                ])?;
                runtime.sample_tokens_mixed(context, &parts, options)?.text
            }
            None => runtime.generate_mixed(prompt, &parts, options)?,
        })
    };
    match (prompt_set, output) {
        (Some(set), Some(output)) => {
            let answers = set
                .prompts
                .iter()
                .enumerate()
                .map(|(index, prompt)| {
                    workflow::progress(format!(
                        "answering prompt {index} of {}",
                        set.prompts.len()
                    ));
                    Ok(serde_json::json!({ "prompt": prompt, "model_output": answer(prompt)? }))
                })
                .collect::<Result<Vec<_>>>()?;
            if let Some(parent) = output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            std::fs::write(&output, serde_json::to_vec_pretty(&answers)?)
                .with_context(|| format!("failed to write {}", output.display()))?;
            println!("{}", output.display());
        }
        _ => {
            let prompt = prompt.context("ster generate needs --prompt or --prompts")?;
            println!("{}", answer(&prompt)?);
        }
    }
    Ok(())
}
