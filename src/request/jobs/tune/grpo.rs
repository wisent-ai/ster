//! The group-relative policy optimization job: the `ster tune grpo` arm over
//! the request API, checkpoints and resume included.

use std::path::Path;

use anyhow::Result;
use serde_json::{Value, json};

use crate::{
    ChatChoice, DeviceChoice, GenerationOptions, GrpoOptions, Precision, PromptSet, Reward, Runtime, UserSimulator,
    lora, tune,
};

use super::super::super::requests::{TuneGrpoRequest, note_precision};
use super::super::tune_flags::{parse_adapter_layers, parse_targets};

/// Mirrors the `ster tune grpo` arm. The reward source is resolved before the
/// policy is loaded, so a reward artifact for the wrong checkpoint is refused
/// before the desktop waits out a policy load to hear it.
pub(in crate::request) fn tune_grpo_job(request: TuneGrpoRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    let source = Reward::parse(&request.reward, &request.model.model, request.model.revision.as_deref(), device)?;
    let precision = Precision::parse(&request.precision)?;
    let user = request
        .user_model
        .as_deref()
        .map(|name| UserSimulator::load(name, request.user_revision.as_deref(), device, precision))
        .transpose()?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat_template = ChatChoice::parse(&request.chat_template)?;
    let chat = runtime.set_chat_template(chat_template)?;
    let prompts = PromptSet::load(Path::new(&request.prompts))?;
    // Built exactly as the CLI builds it, so a run started from one surface
    // resumes from the other.
    let identity = tune::identity(
        &request.model.model,
        request.model.revision.as_deref(),
        &[("prompts", Path::new(&request.prompts))],
        json!(format!("{chat_template:?}")),
        json!(format!("{precision:?}")),
    )?;
    let options = GrpoOptions {
        spec: spec.clone(),
        group: request.group,
        turns: request.turns,
        iterations: request.iterations,
        beta: request.beta,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        generation: GenerationOptions {
            strength: None,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: Some(request.top_p),
            seed: request.seed,
        },
        durable: request.durable.durable(),
        identity,
    };
    let report = tune::grpo(
        &runtime,
        &varmap,
        &prompts,
        &source,
        &request.reward,
        user.as_ref().zip(request.user_model.as_deref()),
        &options,
    )?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}
