//! Requests over an adapter that already exists: merge it into the weights,
//! score a checkpoint with or without it, inspect it.

use serde::Deserialize;

use super::super::defaults::*;
use super::super::{ModelRequest, Validate, require};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneMergeRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// The adapter to fold in; it must be a generation adapter trained for
    /// this exact model.
    #[serde(default)]
    pub(in crate::request) adapter: String,
    /// Directory to write: model.safetensors beside the source's own
    /// config.json and tokenizer.json, plus whichever of
    /// tokenizer_config.json and chat_template.jinja the source published,
    /// which together are what `model` accepts. Those last two are where a
    /// chat template lives, so a source that published one merges to a
    /// checkpoint that still reports `applied` rather than `absent`.
    #[serde(default)]
    pub(in crate::request) output: String,
}

impl Validate for TuneMergeRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune merge")?;
        require(&self.adapter, "tune merge requires an adapter".to_owned())?;
        require(
            &self.output,
            "tune merge requires an output directory".to_owned(),
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneEvaluateRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) examples: String,
    /// A frozen adapter to attach before scoring; omit or leave empty to score
    /// the bare checkpoint, which is the run an adapter is compared against.
    #[serde(default)]
    pub(in crate::request) adapter: Option<String>,
    /// Sequence and batch sizes are the caller's; Ster assumes none.
    pub(in crate::request) max_sequence: usize,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    pub(in crate::request) batch_size: usize,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for TuneEvaluateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("tune evaluate")?;
        require(
            &self.examples,
            "tune evaluate requires an example set".to_owned(),
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TuneInspectRequest {
    #[serde(default)]
    pub(in crate::request) artifact: String,
}

impl Validate for TuneInspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(
            &self.artifact,
            "tune inspect requires an adapter artifact".to_owned(),
        )
    }
}
