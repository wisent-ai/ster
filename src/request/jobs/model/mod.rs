//! Native model creation uses the same library producers as the CLI.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::requests::ModelCreateRequest;

pub(in crate::request) fn fixture_create_job(request: ModelCreateRequest) -> Result<Value> {
    let report =
        ster::fixture::create(Path::new(&request.declaration), Path::new(&request.output))?;
    serde_json::to_value(report).context("serialize the model fixture creation report")
}

pub(in crate::request) fn tokenizer_create_job(request: ModelCreateRequest) -> Result<Value> {
    let report =
        ster::tokenizer::create(Path::new(&request.declaration), Path::new(&request.output))?;
    serde_json::to_value(report).context("serialize the model tokenizer creation report")
}
