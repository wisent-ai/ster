//! The decision operations' requests: answering typed questions about a
//! state, and fitting the temperature that makes those answers honest.

use anyhow::Result;
use serde::Deserialize;

use ster::{ChatChoice, Precision, decide::Request};

use super::{
    ModelRequest, Validate,
    defaults::{default_chat_template, default_precision},
    require,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct DecideRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// The decision document itself — a state and its typed questions — as
    /// the desktop composed it, rather than a path to one on disk. A body
    /// without one does not match the operation's schema.
    pub(in crate::request) request: Request,
    /// A calibration artifact written by `ster calibrate` for this model.
    #[serde(default)]
    pub(in crate::request) calibration: Option<String>,
    /// Option orders per question; `0` shows every option under every letter.
    #[serde(default)]
    pub(in crate::request) permutations: usize,
    /// Whether the response carries per-order detail for every question.
    #[serde(default)]
    pub(in crate::request) explain: bool,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
}

impl Validate for DecideRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("decide")?;
        self.request.validate()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct CalibrateRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// Path to the labelled example set.
    #[serde(default)]
    pub(in crate::request) examples: String,
    /// Where the calibration artifact is written.
    #[serde(default)]
    pub(in crate::request) output: String,
    #[serde(default)]
    pub(in crate::request) permutations: usize,
    /// Equal-width confidence bins the ECE is measured over; required.
    pub(in crate::request) ece_bins: usize,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
}

impl Validate for CalibrateRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("calibrate")?;
        ster::decide::check_bins(self.ece_bins)?;
        require(
            &self.examples,
            "calibrate",
            "examples",
            "the path of a labelled example set",
            "--examples",
        )?;
        require(
            &self.output,
            "calibrate",
            "output",
            "the path the calibration artifact is written to",
            "--output",
        )
    }
}
