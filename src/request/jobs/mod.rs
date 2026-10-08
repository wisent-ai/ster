//! What each operation actually runs. Every job mirrors its CLI arm in the
//! `cli` module: same loads, same workflow call, and the returned document is
//! the same payload the CLI prints.

use anyhow::{Context, Result};
use serde_json::{Value, json};

use std::path::Path;

use crate::{
    ChatChoice, DeviceChoice, GenerationOptions, PairSet, Precision, Runtime, SteeringArtifact,
    TrainingMethod, tune as tune_lib,
    workflow::{self, parse_layers},
};

use super::requests::{
    CompareRequest, EvaluateRequest, ExtractRequest, GenerateRequest, InspectRequest,
    OptimizeRequest, ParityRequest, ProjectRequest, TrainRequest,
};

mod decide;
mod pairs;
mod tune;
mod tune_flags;

pub(super) use decide::{calibrate_job, decide_job};

pub(super) use pairs::{
    pairs_import_job, pairs_inspect_job, pairs_save_job, pairs_synthesize_job,
    workspace_import_pairs_job, workspace_remove_job, workspace_select_job, workspace_show_job,
};
pub(super) use tune::{
    tune_dpo_job, tune_evaluate_job, tune_grpo_job, tune_inspect_job, tune_merge_job,
    tune_reward_job, tune_sft_job,
};

/// Every job mirrors its CLI arm in main.rs: same loads, same workflow call,
/// and the returned document is the same payload the CLI prints.
pub(in crate::request) fn train_job(request: TrainRequest) -> Result<Value> {
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
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
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let selection = workflow::optimize(&runtime, &pair_set, &layers, request.holdout)?;
    selection.artifact.save(Path::new(&request.output))?;
    let mut summary = selection.summary();
    chat.annotate(&mut summary)?;
    Ok(summary)
}

pub(in crate::request) fn evaluate_job(request: EvaluateRequest) -> Result<Value> {
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let artifact = SteeringArtifact::load(Path::new(&request.vector))?;
    tune_lib::warn_on_provenance(Path::new(&request.vector), "direction", &runtime);
    let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
    let mut report = serde_json::to_value(&report)?;
    if let (false, Some(batch), Some(max_sequence)) = (
        request.strengths.is_empty(),
        request.batch_size,
        request.max_sequence,
    ) {
        let options = tune_lib::StrengthOptions {
            strengths: request.strengths,
            batch,
            max_sequence,
        };
        let selection = tune_lib::strengths(&runtime, &pair_set, &artifact, &options)?;
        report
            .as_object_mut()
            .context("an evaluation report is a JSON object")?
            .insert("strength".to_owned(), serde_json::to_value(selection)?);
    }
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
    let artifacts = request
        .steering
        .iter()
        .map(|part| SteeringArtifact::load(Path::new(&part.vector)))
        .collect::<Result<Vec<_>>>()?;
    let parts: Vec<(&SteeringArtifact, f64)> = artifacts
        .iter()
        .zip(request.steering.iter().map(|part| part.strength))
        .collect();
    // An adapter rewrites the projections themselves, so it is attached while
    // the weights are mapped rather than applied per token the way a steering
    // vector is.
    let mut runtime = match request
        .adapter
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        Some(adapter) => Runtime::load_with_adapter_at(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            Path::new(adapter),
            precision,
        )?,
        None => request.model.load_runtime_at(&request.precision)?,
    };
    runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
    for part in &request.steering {
        tune_lib::warn_on_provenance(Path::new(&part.vector), "direction", &runtime);
    }
    let generated = runtime.generate_mixed(
        &request.prompt,
        &parts,
        GenerationOptions {
            strength: None,
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
    runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let input = Path::new(&request.input);
    let output = Path::new(&request.output);
    workflow::extract(&runtime, input, output, &layers)?;
    Ok(json!({"path": request.output}))
}

pub(in crate::request) fn parity_job(request: ParityRequest) -> Result<Value> {
    let runtime = request.model.load_runtime_at(&request.precision)?;
    Ok(serde_json::to_value(workflow::parity(
        &runtime,
        Path::new(&request.input),
    )?)?)
}

pub(in crate::request) fn inspect_job(request: InspectRequest) -> Result<Value> {
    let artifact = SteeringArtifact::load(Path::new(&request.artifact))
        .with_context(|| format!("failed to inspect {}", request.artifact))?;
    Ok(workflow::artifact_summary(&artifact))
}

/// Mirrors `ster vector compare`: the artifacts are read as the CLI reads
/// them, and the report is the document it prints.
pub(in crate::request) fn compare_job(request: CompareRequest) -> Result<Value> {
    let artifacts = request
        .artifacts
        .iter()
        .map(|path| {
            SteeringArtifact::load(Path::new(path)).with_context(|| format!("failed to compare {path}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let options = workflow::CompareOptions {
        layers: request.layers,
        clusters: request.clusters,
    };
    Ok(serde_json::to_value(workflow::compare(&artifacts, &options)?)?)
}

/// Mirrors `ster vector project`: the same reads, the same projection, and
/// the document the CLI prints.
pub(in crate::request) fn project_job(request: ProjectRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let artifact = SteeringArtifact::load(Path::new(&request.vector))?;
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
    tune_lib::warn_on_provenance(Path::new(&request.vector), "direction", &runtime);
    let options = workflow::ProjectOptions {
        layer: request.layer,
        strength: request.strength,
        components: request.components,
    };
    let mut report = serde_json::to_value(workflow::project(&runtime, &pair_set, &artifact, &options)?)?;
    chat.annotate(&mut report)?;
    Ok(report)
}
