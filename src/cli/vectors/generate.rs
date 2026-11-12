//! `ster generate`: text from a model, steered by any number of steering
//! artifacts each at its own strength, or through an adapter, for one prompt
//! or a prompt set.

use std::path::PathBuf;

use anyhow::Result;
use ster::{
    ChatChoice, GenerationOptions, Precision, Runtime, SteeringArtifact,
    failures::{
        file, refuse,
        runtime::{
            encode::TEXT_BLANK,
            generate::{STRENGTHS_UNPAIRED, SYSTEM_EMPTY},
        },
    },
    tune, workflow,
};

use super::super::ModelArgs;

/// What the answers file is called in a refusal of it.
const ANSWERS: &str = "the answers";

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
    // How the flags pair up is checked before any file is read.
    if vector.len() != strength.len() {
        return Err(refuse(
            &STRENGTHS_UNPAIRED,
            &[
                ("vectors", &vector.len().to_string()),
                ("strengths", &strength.len().to_string()),
            ],
        ));
    }
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
    let parts: Vec<(&SteeringArtifact, f64)> =
        artifacts.iter().zip(strength.iter().copied()).collect();
    // The system turn is read here for the same reason: an unreadable or
    // empty file is refused before the checkpoint is paid for.
    let system = match system.as_deref() {
        Some(path) => {
            let shown = path.display().to_string();
            let text = std::fs::read_to_string(path)
                .map_err(|error| file::read::refusal("the system turn", path, &error))?;
            if text.trim().is_empty() {
                return Err(refuse(
                    &SYSTEM_EMPTY,
                    &[("path", &shown), ("bytes", &text.len().to_string())],
                ));
            }
            if let Some(prompt) = prompt.as_deref().filter(|prompt| prompt.trim().is_empty()) {
                return Err(refuse(
                    &TEXT_BLANK,
                    &[("what", "prompt"), ("text", &format!("{prompt:?}"))],
                ));
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
    let options = GenerationOptions {
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
            // The answers' directory is made before the first prompt is
            // answered, so a path no file can be written at is refused
            // before the model's work, not after it.
            if let Some(parent) = output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent).map_err(|error| {
                    let left = "no prompt was answered and nothing was written";
                    file::write::refusal(
                        "creating the directory of",
                        ANSWERS,
                        &output,
                        left,
                        &error,
                    )
                })?;
            }
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
            let text = ster::json::pretty(&answers).into_bytes();
            std::fs::write(&output, text).map_err(|error| {
                let left = format!(
                    "all {} prompts were answered and their answers are lost; a file that was there may be emptied or cut short",
                    answers.len()
                );
                file::write::refusal("writing", ANSWERS, &output, &left, &error)
            })?;
            println!("{}", output.display());
        }
        _ => {
            // `#[arg(long, requires = "output")] prompts` and `#[arg(long,
            // requires = "prompts")] output` leave this arm only with neither
            // given, and `#[arg(long, required_unless_present = "prompts")]
            // prompt` then makes clap refuse a run without --prompt.
            let prompt =
                prompt.expect("clap requires --prompt unless --prompts and --output are given");
            println!("{}", answer(&prompt)?);
        }
    }
    Ok(())
}
