//! Running the adapter-training operations, and the two flag parsers their
//! requests share.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use serde_json::{json, Value};

use std::path::Path;

use crate::{
    lora, tune, ChatChoice, DeviceChoice, DpoLoss, DpoOptions, EvaluateOptions, ExampleSet,
    GrpoOptions, Precision, Reward, Runtime,
    workflow::parse_layers, GenerationOptions, PairSet, PromptSet, RewardHead, RewardOptions,
    SftOptions,

};

use super::super::requests::note_precision;
use super::super::requests::{
    TuneDpoRequest, TuneEvaluateRequest, TuneGrpoRequest, TuneInspectRequest, TuneMergeRequest,
    TuneRewardRequest, TuneSftRequest,
};

/// Mirrors the `ster tune sft` arm: same spec, same `tune::sft`, and the
/// progress lines the trainer writes reach the desktop over the job stream.
pub(in crate::request) fn tune_sft_job(request: TuneSftRequest) -> Result<Value> {
    let device = DeviceChoice::parse(&request.model.device)?;
    let spec = lora::Spec {
        rank: request.rank,
        alpha: request.alpha,
        targets: parse_targets(&request.targets)?,
        layers: parse_adapter_layers(&request.layers)?,
        seed: request.seed,
    };
    // The adapters have to exist before the first forward pass, so the
    // runtime is built from the spec rather than patched after loading; the
    // returned VarMap owns every trainable tensor.
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    let options = SftOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::sft(&runtime, &varmap, &examples, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it, and the encoding it was produced in
    // travels with it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

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
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let pairs = PairSet::load(Path::new(&request.pairs))?;
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
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    // The head is registered at the parameter dtype, never the base dtype: a
    // scalar head is exactly the small trained weight that rounds away in
    // half precision.
    let head =
        RewardHead::fresh(&varmap, runtime.hidden_size(), runtime.device(), runtime.param_dtype())?;
    let pairs = PairSet::load(Path::new(&request.pairs))?;
    let options = RewardOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
    };
    let report = tune::reward(&runtime, &varmap, &head, &pairs, &options)?;
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.reward_artifact(&spec, head.weight(), report.clone())?;
    artifact.save(Path::new(&request.output))?;
    Ok(json!({"path": request.output, "report": report}))
}

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
    let source = Reward::parse(
        &request.reward,
        &request.model.model,
        request.model.revision.as_deref(),
        device,
    )?;
    let precision = Precision::parse(&request.precision)?;
    let (mut runtime, varmap) = Runtime::load_trainable_at(
        &request.model.model,
        request.model.revision.as_deref(),
        device,
        &spec,
        precision,
    )?;
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let prompts = PromptSet::load(Path::new(&request.prompts))?;
    let options = GrpoOptions {
        spec: spec.clone(),
        group: request.group,
        iterations: request.iterations,
        beta: request.beta,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        generation: GenerationOptions {
            strength: 1.0,
            max_new_tokens: request.max_new_tokens,
            temperature: request.temperature,
            top_p: Some(request.top_p),
            seed: request.seed,
        },
    };
    let report = tune::grpo(&runtime, &varmap, &prompts, &source, &request.reward, &options)?;
    // The report is folded into the artifact so a trained adapter always
    // carries the run that produced it.
    let mut report = serde_json::to_value(&report)?;
    chat.annotate(&mut report)?;
    note_precision(&mut report, precision)?;
    let artifact = runtime.adapter_artifact(&spec, report.clone())?;
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
    let adapter = request.adapter.as_deref().filter(|value| !value.trim().is_empty());
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
    let chat = runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    let report = tune::evaluate(
        &runtime,
        &examples,
        adapter.map(Path::new),
        &EvaluateOptions { max_sequence: request.max_sequence, batch: request.batch_size },
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

/// `targets` is a comma-separated projection list, exactly as `--targets` is
/// on the CLI. Repeats collapse and the order follows the request.
fn parse_targets(value: &str) -> Result<Vec<lora::Target>> {
    let mut targets = Vec::new();
    for segment in value.split(',').map(str::trim).filter(|segment| !segment.is_empty()) {
        let target = lora::Target::parse(segment)?;
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    if targets.is_empty() {
        bail!("no targets selected");
    }
    Ok(targets)
}

/// `layers` means what it means everywhere else in Ster, with one difference:
/// `all` cannot be expanded yet. `parse_layers` needs the model's layer count,
/// and the count is only known once the weights are mapped — which happens
/// inside `Runtime::load_trainable`, after the spec exists. An empty layer
/// list is the spec's way of saying every layer, and the loader resolves it
/// against the real count before it builds any adapter.
fn parse_adapter_layers(value: &str) -> Result<Vec<usize>> {
    if value.trim() == "all" {
        return Ok(Vec::new());
    }
    parse_layers(value, usize::MAX)
}
