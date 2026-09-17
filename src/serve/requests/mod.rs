//! What the desktop app may ask for, one struct per endpoint, checked before
//! any model is loaded.
//!
//! The checkpoint half every request carries is here; the rest is grouped the
//! way the endpoints are: steering runs in `vectors`, pair authoring in
//! `pairs`, adapter training in `tune`, and every serde default in
//! `defaults`.

use anyhow::Result;
use serde::Deserialize;

use crate::{DeviceChoice, Precision, Runtime};

use defaults::default_device;

mod defaults;

pub(super) use defaults::note_precision;
mod decide;
mod pairs;
mod tune;
mod vectors;

pub(super) use decide::{CalibrateRequest, DecideRequest};
pub(super) use pairs::{
    PairsInspectRequest, PairsSaveRequest, PairsSynthesizeRequest, WorkspaceImportPairsRequest,
};
pub(super) use tune::{
    TuneDpoRequest, TuneEvaluateRequest, TuneGrpoRequest, TuneInspectRequest, TuneMergeRequest,
    TuneRewardRequest, TuneSftRequest,
};
pub(super) use vectors::{
    EvaluateRequest, ExtractRequest, GenerateRequest, InspectRequest, OptimizeRequest,
    TrainRequest,
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
pub(in crate::serve) struct ModelRequest {
    #[serde(default)]
    pub(in crate::serve) model: String,
    #[serde(default)]
    pub(in crate::serve) revision: Option<String>,
    #[serde(default = "default_device")]
    pub(in crate::serve) device: String,
}

impl ModelRequest {
    fn check(&self, action: &str) -> Result<(), String> {
        require(&self.model, format!("{action} requires a model"))
    }

    /// The shared load. Every handler that maps a checkpoint goes through
    /// here, so `precision` means the same thing on all of them.
    pub(in crate::serve) fn load_runtime_at(&self, precision: &str) -> Result<Runtime> {
        let device = DeviceChoice::parse(&self.device)?;
        Runtime::load_at(
            &self.model,
            self.revision.as_deref(),
            device,
            Precision::parse(precision)?,
        )
    }
}
