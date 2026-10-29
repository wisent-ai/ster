//! What the desktop app may ask for, one struct per operation, checked before
//! any model is loaded.
//!
//! Shared checkpoint identifiers are here; non-tuning inputs are grouped by
//! their owning operation. Tuning decodes the actual typed CLI arguments in
//! `request::tuning` instead of maintaining another schema.

use anyhow::Result;
use serde::Deserialize;

use ster::{DeviceChoice, Precision, Runtime};

use defaults::default_device;

mod defaults;

mod decide;
mod model;
mod pairs;
mod vectors;

pub(super) use decide::{CalibrateRequest, DecideRequest};
pub(super) use model::ModelCreateRequest;
pub(super) use pairs::{
    PairsImportRequest, PairsInspectRequest, PairsMergeRequest, PairsSaveRequest,
    PairsSynthesizeRequest, WorkspaceImportPairsRequest, WorkspacePairSetRequest,
    WorkspaceShowRequest,
};
pub(super) use vectors::{
    AblateRequest, CompareRequest, CurveRequest, EvaluateRequest, ExtractRequest, GenerateRequest,
    InspectRequest, OptimizeRequest, ParityRequest, ProjectRequest, TrainRequest,
};

/// Field-level validation before a job starts streaming. The message is the
/// one-sentence refusal the desktop shows for a malformed request.
pub(super) trait Validate {
    fn validate(&self) -> Result<(), String>;
}

pub(super) fn require(value: &str, sentence: String) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(sentence)
    } else {
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct ModelRequest {
    #[serde(default)]
    pub(in crate::request) model: String,
    #[serde(default)]
    pub(in crate::request) revision: Option<String>,
    #[serde(default = "default_device")]
    pub(in crate::request) device: DeviceChoice,
}

impl ModelRequest {
    fn check(&self, action: &str) -> Result<(), String> {
        require(&self.model, format!("{action} requires a model"))
    }

    /// The shared load. Every handler that maps a checkpoint goes through
    /// here, so `precision` means the same thing on all of them.
    pub(in crate::request) fn load_runtime_at(&self, precision: Precision) -> Result<Runtime> {
        Runtime::load_at(
            &self.model,
            self.revision.as_deref(),
            self.device,
            precision,
        )
    }
}
