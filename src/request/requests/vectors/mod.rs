//! The steering operations' requests: training a direction, choosing one,
//! scoring one, generating with one, and reading raw representations.

use serde::Deserialize;

use super::defaults::*;
use super::{require, ModelRequest, Validate};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct TrainRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) pairs: String,
    #[serde(default)]
    pub(in crate::request) output: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    #[serde(default = "default_method")]
    pub(in crate::request) method: String,
    /// `auto` reads every pair through the model's own chat template when it
    /// publishes one, `off` reads it as raw text. A direction is fitted in
    /// whatever space the pairs were read in and added in whatever space
    /// generation runs in.
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    /// The dtype the base weights are mapped at: `f32`, `f16`, or `bf16`. A
    /// direction is fitted in whatever space the prompts were read in, so two
    /// artifacts trained at different precisions are not interchangeable.
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for TrainRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("train")?;
        require(&self.pairs, "train requires a pairs file".to_owned())?;
        require(&self.output, "train requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct OptimizeRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) pairs: String,
    #[serde(default)]
    pub(in crate::request) output: String,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
}

impl Validate for OptimizeRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("optimize")?;
        require(&self.pairs, "optimize requires a pairs file".to_owned())?;
        require(&self.output, "optimize requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct EvaluateRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) pairs: String,
    #[serde(default)]
    pub(in crate::request) vector: String,
    /// It should match the run that trained the artifact for the same reason
    /// `precision` should.
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for EvaluateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("evaluate")?;
        require(&self.pairs, "evaluate requires a pairs file".to_owned())?;
        require(&self.vector, "evaluate requires a steering artifact".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct GenerateRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) prompt: String,
    #[serde(default)]
    pub(in crate::request) vector: Option<String>,
    /// A frozen LoRA adapter artifact trained for this exact model. Ster
    /// refuses a mismatch rather than steering the wrong residual stream.
    #[serde(default)]
    pub(in crate::request) adapter: Option<String>,
    #[serde(default = "default_strength")]
    pub(in crate::request) strength: f64,
    #[serde(default = "default_max_new_tokens")]
    pub(in crate::request) max_new_tokens: usize,
    #[serde(default)]
    pub(in crate::request) temperature: f64,
    #[serde(default)]
    pub(in crate::request) top_p: Option<f64>,
    #[serde(default = "default_seed")]
    pub(in crate::request) seed: u64,
    /// `auto` renders the prompt through the model's own chat template when
    /// it publishes one, `off` sends raw text. An instruct checkpoint handed
    /// a bare prompt continues it instead of answering it.
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for GenerateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("generate")?;
        require(&self.prompt, "generate requires a prompt".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct ExtractRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) input: String,
    #[serde(default)]
    pub(in crate::request) output: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for ExtractRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("extract")?;
        require(&self.input, "extract requires a prompt input file".to_owned())?;
        require(&self.output, "extract requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct InspectRequest {
    #[serde(default)]
    pub(in crate::request) artifact: String,
}

impl Validate for InspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.artifact, "inspect requires a steering artifact".to_owned())
    }
}
