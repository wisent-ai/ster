//! What each operation actually runs. Every job mirrors its CLI arm in the
//! `cli` module: same loads, same workflow call, and the returned document is
//! the same payload the CLI prints.

use anyhow::{Context, Result};
use serde_json::{json, Value};

use std::path::Path;

use crate::{
    workflow::{self, parse_layers},
    tune as tune_lib, ChatChoice, DeviceChoice, GenerationOptions, PairSet, Precision,
    Runtime,
    SteeringArtifact, TrainingMethod,
};

use super::requests::{
    EvaluateRequest, ExtractRequest, GenerateRequest, InspectRequest, OptimizeRequest,
    TrainRequest,
};

mod decide;
mod pairs;
mod tune;

pub(super) use decide::{calibrate_job, decide_job};

pub(super) use pairs::{
    pairs_inspect_job, pairs_save_job, pairs_synthesize_job, workspace_import_pairs_job,
};
pub(super) use tune::{
    tune_dpo_job, tune_evaluate_job, tune_grpo_job, tune_inspect_job, tune_merge_job,
    tune_reward_job, tune_sft_job,
};

/// Every job mirrors its CLI arm in main.rs: same loads, same workflow call,
/// and the returned document is the same payload the CLI prints.
pub(in crate::request) fn train_job(request: TrainRequest) -> Result<Value> {
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let method = TrainingMethod::parse(&request.method)?;
    let artifact = workflow::train(&runtime, &pair_set, &layers, method)?;
    artifact.save(Path::new(&request.output))?;
    let mut summary = workflow::artifact_summary(&artifact);
    chat.annotate(&mut summary)?;
    Ok(summary)
}

pub(in crate::request) fn optimize_job(request: OptimizeRequest) -> Result<Value> {
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let selection = workflow::optimize(&runtime, &pair_set, &layers)?;
    selection.artifact.save(Path::new(&request.output))?;
    let mut summary = selection.summary();
    chat.annotate(&mut summary)?;
    Ok(summary)
}

pub(in crate::request) fn evaluate_job(request: EvaluateRequest) -> Result<Value> {
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let artifact = SteeringArtifact::load(Path::new(&request.vector))?;
    tune_lib::warn_on_provenance(Path::new(&request.vector), "direction", &runtime);
    let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    Ok(report)
}

pub(in crate::request) fn generate_job(request: GenerateRequest) -> Result<Value> {
    let precision = Precision::parse(&request.precision)?;
    // Both documents are read before a single weight is mapped, so the two
    // halves of the wrong-document refusal cost the same. An adapter was
    // already refused this early because it is attached during the load; a
    // steering vector was not, and the wrong file there paid for a full
    // checkpoint load before being told.
    let vector = request.vector.as_deref().filter(|value| !value.trim().is_empty());
    let artifact = vector.map(|value| SteeringArtifact::load(Path::new(value))).transpose()?;
    // An adapter rewrites the projections themselves, so it is attached while
    // the weights are mapped rather than applied per token the way a steering
    // vector is.
    let mut runtime = match request.adapter.as_deref().filter(|value| !value.trim().is_empty()) {
        Some(adapter) => Runtime::load_with_adapter_at(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            Path::new(adapter),
            precision,
        )?,
        None => request.model.load_runtime_at(&request.precision)?,
    };
    runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    if let Some(vector) = vector {
        tune_lib::warn_on_provenance(Path::new(vector), "direction", &runtime);
    }
    let generated = runtime.generate(
        &request.prompt,
        artifact.as_ref(),
        GenerationOptions {
            strength: request.strength,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
        },
    )?;
    Ok(json!({"text": generated}))
}

pub(in crate::request) fn extract_job(request: ExtractRequest) -> Result<Value> {
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    workflow::extract(&runtime, input, output, &layers)?;
    Ok(json!({"path": request.output}))
}

pub(in crate::request) fn inspect_job(request: InspectRequest) -> Result<Value> {
    let artifact = SteeringArtifact::load(Path::new(&request.artifact))
        .with_context(|| format!("failed to inspect {}", request.artifact))?;
    Ok(workflow::artifact_summary(&artifact))
}

