//! Running the adapter-training operations.

use anyhow::{Context, Result};
use candle_core::Device;
use serde_json::{Value, json};

use std::path::Path;

use crate::{
    ChatChoice, DeviceChoice, DpoLoss, DpoOptions, EvaluateOptions, ExampleSet, PairSet, Precision, RewardHead,
    RewardOptions, Runtime, lora, tune,
};

mod grpo;
mod sft;

pub(in crate::request) use grpo::tune_grpo_job;
pub(in crate::request) use sft::tune_sft_job;

use super::super::requests::note_precision;
use super::super::requests::{
    TuneDpoRequest, TuneEvaluateRequest, TuneInspectRequest, TuneMergeRequest, TuneRewardRequest,
};
use super::tune_flags::{parse_adapter_layers, parse_targets};

/// Mirrors the `ster tune dpo` arm. One runtime is loaded: the reference the
/// objective measures against is the same weights with the adapters skipped.
pub(in crate::request) fn tune_dpo_job(request: TuneDpoRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat_template = ChatChoice::parse(&request.chat_template)?;
    let chat = runtime.set_chat_template(chat_template)?;
    let pairs = PairSet::load(Path::new(&request.pairs))?;
    // Built exactly as the CLI builds it, so a run started from one surface
    // resumes from the other.
    let identity = tune::identity(
        &request.model.model,
        request.model.revision.as_deref(),
        &[("pairs", Path::new(&request.pairs))],
        json!(format!("{chat_template:?}")),
        json!(format!("{precision:?}")),
    )?;
    let options = DpoOptions {
        spec: spec.clone(),
        loss: DpoLoss::parse(&request.loss)?,
        beta: request.beta,
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
        durable: request.durable.durable(),
        identity,
    };
    let report = tune::dpo(&runtime, &varmap, &pairs, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

/// Mirrors the `ster tune reward` arm. The head is registered in the same
/// VarMap the adapters live in, so one optimizer steps the pair and the
/// artifact carries both.
pub(in crate::request) fn tune_reward_job(request: TuneRewardRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat_template = ChatChoice::parse(&request.chat_template)?;
    let chat = runtime.set_chat_template(chat_template)?;
    // The head is registered at the parameter dtype, never the base dtype: a
    // scalar head is exactly the small trained weight that rounds away in
    // half precision.
    let head = RewardHead::fresh(
        &varmap,
        runtime.hidden_size(),
        runtime.device(),
        runtime.param_dtype(),
    )?;
    let pairs = PairSet::load(Path::new(&request.pairs))?;
    // Built exactly as the CLI builds it, so a run started from one surface
    // resumes from the other.
    let identity = tune::identity(
        &request.model.model,
        request.model.revision.as_deref(),
        &[("pairs", Path::new(&request.pairs))],
        json!(format!("{chat_template:?}")),
        json!(format!("{precision:?}")),
    )?;
    let options = RewardOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
        durable: request.durable.durable(),
        identity,
    };
    let report = tune::reward(&runtime, &varmap, &head, &pairs, &options)?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.reward_artifact(&spec, head.weight(), report.clone())?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

/// Mirrors the `ster tune merge` arm. No device and no runtime: merging
/// rewrites tensors and never runs the model.
pub(in crate::request) fn tune_merge_job(request: TuneMergeRequest) -> Result<Value> {
    let report = tune::merge(
        &request.model.model,
        request.model.revision.as_deref(),
        Path::new(&request.adapter),
        Path::new(&request.output),
    )?;
    Ok(json!({ "report": report }))
}

/// Mirrors the `ster tune evaluate` arm: no optimizer, no artifact written,
/// and the same document the CLI prints.
pub(in crate::request) fn tune_evaluate_job(request: TuneEvaluateRequest) -> Result<Value> {
    let adapter = request
        .adapter
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    let precision = Precision::parse(&request.precision)?;
    let mut runtime = match adapter {
        Some(adapter) => Runtime::load_with_adapter_at(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            Path::new(adapter),
            precision,
        )?,
        None => Runtime::load_at(
            &request.model.model,
            request.model.revision.as_deref(),
            DeviceChoice::parse(&request.model.device)?,
            precision,
        )?,
    };
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?)?;
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    let report = tune::evaluate(
        &runtime,
        &examples,
        adapter.map(Path::new),
        &EvaluateOptions {
            max_sequence: request.max_sequence,
            batch: request.batch_size,
        },
    )?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    Ok(report)
}

pub(in crate::request) fn tune_inspect_job(request: TuneInspectRequest) -> Result<Value> {
    // Inspection reads the adapter document alone: no model is loaded, so the
    // tensors land on the CPU whatever trained them.
    let artifact = lora::Artifact::load(Path::new(&request.artifact), &Device::Cpu)
        .with_context(|| format!("failed to inspect {}", request.artifact))?;
    Ok(tune::inspect(&artifact))
}
