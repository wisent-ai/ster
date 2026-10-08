//! The steering operations' requests: training a direction, choosing one,
//! scoring one, generating with one, and reading raw representations.

use serde::Deserialize;

use super::defaults::*;
use super::{ModelRequest, Validate, require};

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
        self.model.check("vector/train")?;
        require(&self.pairs, "vector/train requires a pairs file".to_owned())?;
        require(&self.output, "vector/train requires an output path".to_owned())
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
    /// Fraction of the pairs held out to rank candidates on; required.
    pub(in crate::request) holdout: f64,
}

impl Validate for OptimizeRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("vector/optimize")?;
        require(&self.pairs, "vector/optimize requires a pairs file".to_owned())?;
        require(&self.output, "vector/optimize requires an output path".to_owned())
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
    /// Strengths to measure the artifact at, as `--strengths` takes them;
    /// empty measures none.
    #[serde(default)]
    pub(in crate::request) strengths: Vec<f64>,
    /// Pairs per forward pass while measuring strengths; required with them.
    #[serde(default)]
    pub(in crate::request) batch_size: Option<usize>,
    /// Longest pair side measured, in tokens; required with strengths.
    #[serde(default)]
    pub(in crate::request) max_sequence: Option<usize>,
}

impl Validate for EvaluateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("vector/evaluate")?;
        require(&self.pairs, "vector/evaluate requires a pairs file".to_owned())?;
        require(
            &self.vector,
            "vector/evaluate requires a steering artifact".to_owned(),
        )?;
        if self.strengths.is_empty() {
            if self.batch_size.is_some() || self.max_sequence.is_some() {
                return Err(
                    "vector/evaluate takes batchSize and maxSequence only with strengths".to_owned(),
                );
            }
            return Ok(());
        }
        if self.batch_size.is_none() {
            return Err("vector/evaluate with strengths requires batchSize; Ster assumes none".to_owned());
        }
        if self.max_sequence.is_none() {
            return Err(
                "vector/evaluate with strengths requires maxSequence; Ster assumes none".to_owned(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct GenerateRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) prompt: String,
    /// Steering artifacts added during generation, each at its own
    /// strength; empty generates unsteered. Token budget, temperature and
    /// seed are the caller's; Ster assumes none. Temperature zero is argmax.
    #[serde(default)]
    pub(in crate::request) steering: Vec<SteeringPart>,
    /// A frozen LoRA adapter artifact trained for this exact model. Ster
    /// refuses a mismatch rather than steering the wrong residual stream.
    #[serde(default)]
    pub(in crate::request) adapter: Option<String>,
    pub(in crate::request) max_new_tokens: usize,
    pub(in crate::request) temperature: f64,
    /// Nucleus mass; absent samples from the whole distribution.
    #[serde(default)]
    pub(in crate::request) top_p: Option<f64>,
    pub(in crate::request) seed: u64,
    /// `auto` renders the prompt through the model's own chat template when
    /// it publishes one, `off` sends raw text. An instruct checkpoint handed
    /// a bare prompt continues it instead of answering it.
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

/// One steering artifact and the scale it is added at. The strength has no
/// default: Ster assumes no steering scale.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct SteeringPart {
    pub(in crate::request) vector: String,
    pub(in crate::request) strength: f64,
}

impl Validate for GenerateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("generate")?;
        for part in &self.steering {
            require(&part.vector, "generate names a steering part with no vector".to_owned())?;
        }
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
        require(
            &self.input,
            "extract requires a prompt input file".to_owned(),
        )?;
        require(&self.output, "extract requires an output path".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct ParityRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    /// The parity file `ster parity --input` reads.
    #[serde(default)]
    pub(in crate::request) input: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for ParityRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("parity")?;
        require(
            &self.input,
            "parity requires a parity input file".to_owned(),
        )
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
        require(
            &self.artifact,
            "vector/inspect requires a steering artifact".to_owned(),
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct CompareRequest {
    /// Steering artifacts fitted on one model, two or more.
    pub(in crate::request) artifacts: Vec<String>,
    /// Layers every artifact carries; all they share when absent.
    #[serde(default)]
    pub(in crate::request) layers: Option<Vec<usize>>,
    /// Groups the artifacts are cut into, at most their count.
    pub(in crate::request) clusters: std::num::NonZeroUsize,
}

impl Validate for CompareRequest {
    fn validate(&self) -> Result<(), String> {
        for artifact in &self.artifacts {
            require(artifact, "vector/compare names an empty artifact path".to_owned())?;
        }
        match self.artifacts.as_slice() {
            [_, _, ..] => Ok(()),
            _ => Err(format!(
                "vector/compare needs at least two artifacts and got {}",
                self.artifacts.len()
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct ProjectRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) pairs: String,
    #[serde(default)]
    pub(in crate::request) vector: String,
    /// The layer every side is read at; required.
    pub(in crate::request) layer: usize,
    /// The strength the artifact is added at; required, Ster assumes none.
    pub(in crate::request) strength: f64,
    /// Principal components to project onto; required.
    pub(in crate::request) components: std::num::NonZeroUsize,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for ProjectRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("vector/project")?;
        require(&self.pairs, "vector/project requires a pairs file".to_owned())?;
        require(&self.vector, "vector/project requires a steering artifact".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct CurveRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) pairs: String,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    /// caa, pca or logistic; required.
    pub(in crate::request) method: String,
    /// Fraction of the pairs held out to score on; required.
    pub(in crate::request) holdout: f64,
    /// Numbers of pairs to fit on; required.
    pub(in crate::request) sizes: Vec<std::num::NonZeroUsize>,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
}

impl Validate for CurveRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("vector/curve")?;
        require(&self.pairs, "vector/curve requires a pairs file".to_owned())?;
        if self.sizes.is_empty() {
            return Err("vector/curve requires at least one size to fit on".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct AblateRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) vector: String,
    /// Share of the direction removed from every write; required.
    pub(in crate::request) strength: f64,
    #[serde(default)]
    pub(in crate::request) output: String,
}

impl Validate for AblateRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("vector/ablate")?;
        require(&self.vector, "vector/ablate requires a steering artifact".to_owned())?;
        require(&self.output, "vector/ablate requires an output directory".to_owned())
    }
}
