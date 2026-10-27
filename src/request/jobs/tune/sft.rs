//! The supervised fine-tuning job: the `ster tune sft` arm over the request
//! API, checkpoints and resume included.

use std::path::Path;

use anyhow::Result;
use serde_json::{Value, json};

use crate::{ChatChoice, DeviceChoice, ExampleSet, Precision, Runtime, SftOptions, lora, tune};

use super::super::super::requests::{TuneSftRequest, note_precision};
use super::super::tune_flags::{parse_adapter_layers, parse_targets};

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
    let chat_template = ChatChoice::parse(&request.chat_template)?;
    let chat = runtime.set_chat_template(chat_template)?;
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    // Built exactly as the CLI builds it, so a run started from one surface
    // resumes from the other.
    let identity = tune::identity(
        &request.model.model,
        request.model.revision.as_deref(),
        &[("examples", Path::new(&request.examples))],
        json!(format!("{chat_template:?}")),
        json!(format!("{precision:?}")),
    )?;
    let durable = request.durable.durable();
    let options = SftOptions {
        spec: spec.clone(),
        epochs: request.epochs,
        learning_rate: request.learning_rate,
        accumulation: request.accumulation,
        batch: request.batch_size,
        warmup_steps: request.warmup_steps,
        max_sequence: request.max_sequence,
        seed: request.seed,
        durable,
        identity,
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
