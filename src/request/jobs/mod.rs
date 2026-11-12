//! Non-tuning native operations. Tuning runs the actual typed CLI operation
//! through `request::tuning`; it has no mirrored job implementation here.

use anyhow::Result;
use serde_json::{Value, json};

use std::path::Path;

use ster::{
    GenerationOptions, PairSet, Runtime, SteeringArtifact, tune as tune_lib,
    workflow::{self, parse_layers},
};

use super::requests::{
    AblateRequest, CompareRequest, CurveRequest, EvaluateRequest, ExtractRequest, GenerateRequest,
    InspectRequest, OptimizeRequest, ParityRequest, ProjectRequest, ScoreRequest, TrainRequest,
};

mod decide;
mod errors;
mod model;
mod pairs;

pub(super) use decide::{calibrate_job, decide_job};
pub(super) use errors::{errors_list_job, errors_show_job};
pub(super) use model::{fixture_create_job, tokenizer_create_job};

pub(super) use pairs::{
    pairs_import_job, pairs_inspect_job, pairs_merge_job, pairs_save_job, pairs_synthesize_job,
    workspace_import_pairs_job, workspace_remove_job, workspace_select_job, workspace_show_job,
};

/// Every job mirrors its CLI arm in main.rs: same loads, same workflow call,
/// and the returned document is the same payload the CLI prints.
pub(in crate::request) fn train_job(request: TrainRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let mut runtime = request.model.load_runtime_at(request.precision)?;
    let chat = runtime.set_chat_template(request.chat_template)?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let artifact = workflow::train(&runtime, &pair_set, &layers, request.method)?;
    artifact.save(Path::new(&request.output))?;
    let mut summary = workflow::artifact_summary(&artifact);
    chat.annotate(&mut summary)?;
    Ok(summary)
}

pub(in crate::request) fn optimize_job(request: OptimizeRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    workflow::split_holdout(&pair_set, request.holdout)?;
    let mut runtime = request.model.load_runtime_at(request.precision)?;
    let chat = runtime.set_chat_template(request.chat_template)?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let selection = workflow::optimize(&runtime, &pair_set, &layers, request.holdout)?;
    selection.artifact.save(Path::new(&request.output))?;
    let mut summary = selection.summary();
    chat.annotate(&mut summary)?;
    Ok(summary)
}

pub(in crate::request) fn evaluate_job(request: EvaluateRequest) -> Result<Value> {
    // `EvaluateRequest::validate` refused strengths given without batchSize
    // or maxSequence before this job ran; strengths come with both.
    let strength = match (
        request.strengths.is_empty(),
        request.batch_size,
        request.max_sequence,
    ) {
        (false, Some(batch), Some(max_sequence)) => Some(tune_lib::StrengthOptions {
            strengths: request.strengths,
            batch,
            max_sequence,
        }),
        _ => None,
    };
    if let Some(options) = &strength {
        options.check()?;
    }
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let mut runtime = request.model.load_runtime_at(request.precision)?;
    let chat = runtime.set_chat_template(request.chat_template)?;
    let artifact = SteeringArtifact::load(Path::new(&request.vector))?;
    tune_lib::warn_on_provenance(Path::new(&request.vector), "direction", &runtime);
    let report = workflow::evaluate(&runtime, &pair_set, &artifact)?;
    let mut report = ster::json::value(&report);
    if let Some(options) = strength {
        let selection = tune_lib::strengths(&runtime, &pair_set, &artifact, &options)?;
        // `json::value(&report)` of `#[derive(Serialize)] pub struct
        // EvaluationReport`, a struct with named fields, is an object.
        report
            .as_object_mut()
            .expect("an evaluation report is a struct with named fields")
            .insert("strength".to_owned(), ster::json::value(selection));
    }
    chat.annotate(&mut report)?;
    Ok(report)
}

pub(in crate::request) fn generate_job(request: GenerateRequest) -> Result<Value> {
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
            request.model.device,
            Path::new(adapter),
            request.precision,
        )?,
        None => request.model.load_runtime_at(request.precision)?,
    };
    runtime.set_chat_template(request.chat_template)?;
    for part in &request.steering {
        tune_lib::warn_on_provenance(Path::new(&part.vector), "direction", &runtime);
    }
    let generated = runtime.generate_mixed(
        &request.prompt,
        &parts,
        GenerationOptions {
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
        },
    )?;
    Ok(json!({"text": generated}))
}

pub(in crate::request) fn extract_job(request: ExtractRequest) -> Result<Value> {
    let prompts = workflow::PromptSet::load(Path::new(&request.input))?;
    let mut runtime = request.model.load_runtime_at(request.precision)?;
    runtime.set_chat_template(request.chat_template)?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    workflow::extract(&runtime, &prompts, Path::new(&request.output), &layers)?;
    Ok(json!({"path": request.output}))
}

pub(in crate::request) fn parity_job(request: ParityRequest) -> Result<Value> {
    let runtime = request.model.load_runtime_at(request.precision)?;
    Ok(ster::json::value(workflow::parity(
        &runtime,
        Path::new(&request.input),
    )?))
}

pub(in crate::request) fn inspect_job(request: InspectRequest) -> Result<Value> {
    let artifact = SteeringArtifact::load(Path::new(&request.artifact))?;
    Ok(workflow::artifact_summary(&artifact))
}

/// Mirrors `ster vector compare`: the artifacts are read as the CLI reads
/// them, and the report is the document it prints.
pub(in crate::request) fn compare_job(request: CompareRequest) -> Result<Value> {
    workflow::check_counts(&request.artifacts, request.clusters)?;
    let artifacts = request
        .artifacts
        .iter()
        .map(|path| SteeringArtifact::load(Path::new(path)))
        .collect::<Result<Vec<_>>>()?;
    let options = workflow::CompareOptions {
        layers: request.layers,
        clusters: request.clusters,
    };
    Ok(ster::json::value(workflow::compare(&artifacts, &options)?))
}

/// Mirrors `ster vector project`: the same reads, the same projection, and
/// the document the CLI prints.
pub(in crate::request) fn project_job(request: ProjectRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    let artifact = SteeringArtifact::load(Path::new(&request.vector))?;
    let mut runtime = request.model.load_runtime_at(request.precision)?;
    let chat = runtime.set_chat_template(request.chat_template)?;
    tune_lib::warn_on_provenance(Path::new(&request.vector), "direction", &runtime);
    let options = workflow::ProjectOptions {
        layer: request.layer,
        strength: request.strength,
        components: request.components,
    };
    let mut report =
        ster::json::value(workflow::project(&runtime, &pair_set, &artifact, &options)?);
    chat.annotate(&mut report)?;
    Ok(report)
}

/// Mirrors `ster vector score`: the same read, the same projection, and the
/// document the CLI prints.
pub(in crate::request) fn score_job(request: ScoreRequest) -> Result<Value> {
    workflow::check_text(&request.text, "the request's text")?;
    let artifact = SteeringArtifact::load(Path::new(&request.vector))?;
    let mut runtime = request.model.load_runtime_at(request.precision)?;
    let chat = runtime.set_chat_template(request.chat_template)?;
    tune_lib::warn_on_provenance(Path::new(&request.vector), "direction", &runtime);
    let mut report = ster::json::value(workflow::score(
        &runtime,
        &artifact,
        &request.text,
        request.layer,
    )?);
    chat.annotate(&mut report)?;
    Ok(report)
}

/// Mirrors `ster vector curve`: the same capture, the same fits, and the
/// document the CLI prints.
pub(in crate::request) fn curve_job(request: CurveRequest) -> Result<Value> {
    let pair_set = PairSet::load(Path::new(&request.pairs))?;
    workflow::check_sizes(&pair_set, request.holdout, &request.sizes, request.method)?;
    let mut runtime = request.model.load_runtime_at(request.precision)?;
    let chat = runtime.set_chat_template(request.chat_template)?;
    let layers = parse_layers(&request.layers, runtime.layer_count())?;
    let report = workflow::curve(
        &runtime,
        &pair_set,
        &layers,
        request.method,
        request.holdout,
        &request.sizes,
    )?;
    let mut report = ster::json::value(report);
    chat.annotate(&mut report)?;
    Ok(report)
}

/// Mirrors `ster vector ablate`: the same rewrite, and the report the CLI
/// prints. Like `tune/merge`, no decoder is built.
pub(in crate::request) fn ablate_job(request: AblateRequest) -> Result<Value> {
    let report = tune_lib::ablate(
        &request.model.model,
        request.model.revision.as_deref(),
        Path::new(&request.vector),
        request.strength,
        Path::new(&request.output),
    )?;
    Ok(ster::json::value(report))
}
