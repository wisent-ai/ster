//! The decision operations' requests: answering typed questions about a
//! state, and fitting the temperature that makes those answers honest.

use serde::Deserialize;

use crate::decide::Request;

use super::{
    defaults::{default_chat_template, default_precision},
    require, ModelRequest, Validate,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct DecideRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// The decision document itself — a state and its typed questions — as
    /// the desktop composed it, rather than a path to one on disk.
    pub(in crate::request) request: Option<Request>,
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
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for DecideRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("decide")?;
        let Some(request) = &self.request else {
            return Err("decide requires a request with a state and questions".to_owned());
        };
        request.validate().map_err(|error| format!("{error:#}"))
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
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for CalibrateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("calibrate")?;
        require(&self.examples, "calibrate requires an examples path".to_owned())?;
        require(&self.output, "calibrate requires an output path".to_owned())
    }
}
