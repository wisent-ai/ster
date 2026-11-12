//! The requests of the steering operations that read trained artifacts:
//! inspecting and comparing them, projecting pairs onto one, a learning
//! curve, and an ablation.

use anyhow::Result;
use serde::Deserialize;

use super::super::{ModelRequest, Validate, defaults::*, require};
use super::{ARTIFACT, PAIRS};
use ster::failures::request::too_few;
use ster::{ChatChoice, Precision, TrainingMethod};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct InspectRequest {
    #[serde(default)]
    pub(in crate::request) artifact: String,
}

impl Validate for InspectRequest {
    fn validate(&self) -> Result<()> {
        require(
            &self.artifact,
            "vector/inspect",
            "artifact",
            ARTIFACT,
            "the ARTIFACT argument of `ster vector inspect`",
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
    fn validate(&self) -> Result<()> {
        const ARGUMENT: &str = "the ARTIFACT arguments of `ster vector compare`";
        for (index, artifact) in self.artifacts.iter().enumerate() {
            require(artifact, "vector/compare", &format!("artifacts[{index}]"), ARTIFACT, ARGUMENT)?;
        }
        if self.artifacts.len() < 2 {
            return Err(too_few(
                "vector/compare",
                "artifacts",
                "steering artifacts",
                2,
                self.artifacts.len(),
                ARGUMENT,
            ));
        }
        Ok(())
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
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
}

impl Validate for ProjectRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("vector/project")?;
        require(&self.pairs, "vector/project", "pairs", PAIRS, "--pairs")?;
        require(&self.vector, "vector/project", "vector", ARTIFACT, "--vector")
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
    pub(in crate::request) method: TrainingMethod,
    /// Fraction of the pairs held out to score on; required.
    pub(in crate::request) holdout: f64,
    /// Numbers of pairs to fit on; required.
    pub(in crate::request) sizes: Vec<std::num::NonZeroUsize>,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
}

impl Validate for CurveRequest {
    fn validate(&self) -> Result<()> {
        self.model.check("vector/curve")?;
        require(&self.pairs, "vector/curve", "pairs", PAIRS, "--pairs")?;
        if self.sizes.is_empty() {
            return Err(too_few("vector/curve", "sizes", "size to fit on", 1, 0, "--sizes"));
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
    fn validate(&self) -> Result<()> {
        self.model.check("vector/ablate")?;
        require(&self.vector, "vector/ablate", "vector", ARTIFACT, "--vector")?;
        require(
            &self.output,
            "vector/ablate",
            "output",
            "the directory the rewritten checkpoint is written to",
            "--output",
        )
    }
}
