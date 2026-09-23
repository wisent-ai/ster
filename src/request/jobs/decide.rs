//! The decision jobs: the same loads and the same library calls as `ster
//! decide` and `ster calibrate`, returning the documents those print.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::{
    decide::{self, Calibration, ExampleSet, RAW_TEMPERATURE},
    ChatChoice, DecideOptions,
};

use super::super::requests::{CalibrateRequest, DecideRequest};

pub(in crate::request) fn decide_job(request: DecideRequest) -> Result<Value> {
    let document = request.request.expect("validated request");
    // The calibration is read before a single weight is mapped, so one
    // fitted for another model is refused in milliseconds rather than after
    // a full checkpoint load.
    let calibration = request
        .calibration
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|path| Calibration::load(Path::new(path)).map(|loaded| (path, loaded)))
        .transpose()?;
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    let mut options = DecideOptions {
        permutations: request.permutations,
        temperature: RAW_TEMPERATURE,
        explain: request.explain,
    };
    if let Some((path, loaded)) = &calibration {
        loaded.check_model(Path::new(path), &runtime.model_id)?;
        options.temperature = loaded.temperature;
    }
    runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let response =
        decide::decide(&runtime, &document, options, calibration.as_ref().map(|(path, _)| *path))?;
    Ok(serde_json::to_value(response)?)
}

pub(in crate::request) fn calibrate_job(request: CalibrateRequest) -> Result<Value> {
    let examples = ExampleSet::load(Path::new(&request.examples))?;
    let mut runtime = request.model.load_runtime_at(&request.precision)?;
    runtime.set_chat_template(ChatChoice::parse(&request.chat_template)?);
    let options = DecideOptions {
        permutations: request.permutations,
        temperature: RAW_TEMPERATURE,
        explain: false,
    };
    let calibration = decide::calibrate(&runtime, &examples, options)?;
    calibration.save(Path::new(&request.output))?;
    Ok(serde_json::to_value(calibration)?)
}
