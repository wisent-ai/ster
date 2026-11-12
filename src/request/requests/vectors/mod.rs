//! The steering operations' requests: training a direction, choosing one,
//! scoring one, generating with one, and reading raw representations.

use anyhow::Result;
use serde::Deserialize;

use super::{ModelRequest, Validate, defaults::*, require};
use ster::failures::request::unpaired;

/// What a request's `pairs` field holds.
const PAIRS: &str = "the path of a pair set";
/// What a request's `vector` field holds.
const ARTIFACT: &str = "the path of a steering artifact";
/// What a request's `output` field holds where it writes an artifact.
const ARTIFACT_OUT: &str = "the path the steering artifact is written to";
use ster::{ChatChoice, Precision, TrainingMethod};

mod artifacts;
mod score;

pub(in crate::request) use artifacts::{
    AblateRequest, CompareRequest, CurveRequest, InspectRequest, ProjectRequest,
};
pub(in crate::request) use score::ScoreRequest;

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
    pub(in crate::request) method: TrainingMethod,
    /// `auto` reads every pair through the model's own chat template when it
    /// publishes one, `off` reads it as raw text. A direction is fitted in
    /// whatever space the pairs were read in and added in whatever space
    /// generation runs in.
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: ChatChoice,
    /// The dtype the base weights are mapped at: `f32`, `f16`, or `bf16`. A
    /// direction is fitted in whatever space the prompts were read in, so two
    /// artifacts trained at different precisions are not interchangeable.
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
}

impl Validate for TrainRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("vector/train")?;
        require(&self.pairs, "vector/train", "pairs", PAIRS, "--pairs")?;
        require(&self.output, "vector/train", "output", ARTIFACT_OUT, "--output")
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
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
    #[serde(default = "default_layers")]
    pub(in crate::request) layers: String,
    /// Fraction of the pairs held out to rank candidates on; required.
    pub(in crate::request) holdout: f64,
}

impl Validate for OptimizeRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("vector/optimize")?;
        require(&self.pairs, "vector/optimize", "pairs", PAIRS, "--pairs")?;
        require(&self.output, "vector/optimize", "output", ARTIFACT_OUT, "--output")
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
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
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
    fn validate(&self) -> Result<()> {
        const OPERATION: &str = "vector/evaluate";
        self.model.check(OPERATION)?;
        require(&self.pairs, OPERATION, "pairs", PAIRS, "--pairs")?;
        require(&self.vector, OPERATION, "vector", ARTIFACT, "--vector")?;
        if self.strengths.is_empty() {
            if let Some(batch) = self.batch_size {
                return Err(unpaired(OPERATION, "batchSize", &batch.to_string(), "strengths"));
            }
            if let Some(longest) = self.max_sequence {
                return Err(unpaired(OPERATION, "maxSequence", &longest.to_string(), "strengths"));
            }
            return Ok(());
        }
        let strengths = format!("{:?}", self.strengths);
        if self.batch_size.is_none() {
            return Err(unpaired(OPERATION, "strengths", &strengths, "batchSize"));
        }
        if self.max_sequence.is_none() {
            return Err(unpaired(OPERATION, "strengths", &strengths, "maxSequence"));
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
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
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
    fn validate(&self) -> Result<()> {
        self.model.check("generate")?;
        for (index, part) in self.steering.iter().enumerate() {
            require(
                &part.vector,
                "generate",
                &format!("steering[{index}].vector"),
                ARTIFACT,
                "--vector",
            )?;
        }
        require(&self.prompt, "generate", "prompt", "the text the model continues", "--prompt")
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
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
}

impl Validate for ExtractRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("extract")?;
        require(
            &self.input,
            "extract",
            "input",
            "the path of a prompt set ({\"prompts\": [...]})",
            "--input",
        )?;
        require(&self.output, "extract", "output", "the path the activations are written to", "--output")
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
    pub(in crate::request) precision: Precision,
}

impl Validate for ParityRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("parity")?;
        require(&self.input, "parity", "input", "the path of a parity records file", "--input")
    }
}
