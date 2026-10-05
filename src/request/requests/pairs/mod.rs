//! The pair-authoring operations' requests: importing a set into the
//! workspace, auditing one, saving one, and writing one with a model.

use serde::Deserialize;

use super::defaults::*;
use super::{require, ModelRequest, Validate};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct WorkspaceImportPairsRequest {
    #[serde(default)]
    pub(in crate::request) source: String,
    #[serde(default)]
    pub(in crate::request) name: Option<String>,
}

impl Validate for WorkspaceImportPairsRequest {
    fn validate(&self) -> Result<(), String> {
        require(
            &self.source,
            "workspace import-pairs requires a source path".to_owned(),
        )
    }
}

/// `workspace/show`: no fields; the answer is `ster workspace show`'s.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::request) struct WorkspaceShowRequest {}

impl Validate for WorkspaceShowRequest {
    fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

/// `workspace/select` and `workspace/remove`: one imported set by its
/// workspace name.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct WorkspacePairSetRequest {
    #[serde(default)]
    pub(in crate::request) id: String,
}

impl Validate for WorkspacePairSetRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.id, "a workspace pair-set request requires the set's id".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct PairsInspectRequest {
    #[serde(default)]
    pub(in crate::request) pairs: String,
    /// The deduplication and refusal settings are the caller's; Ster assumes
    /// none, and a body without one is refused by name.
    pub(in crate::request) dedupe_bits: u32,
    pub(in crate::request) dedupe_bands: u32,
    pub(in crate::request) refusal_threshold: f32,
    pub(in crate::request) unbalanced_ratio: f64,
}

impl Validate for PairsInspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(
            &self.pairs,
            "pairs inspect requires a pairs file".to_owned(),
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct PairsSaveEntry {
    #[serde(default)]
    pub(in crate::request) positive: String,
    #[serde(default)]
    pub(in crate::request) negative: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct PairsSaveRequest {
    #[serde(default)]
    pub(in crate::request) path: String,
    #[serde(default)]
    pub(in crate::request) trait_name: String,
    #[serde(default)]
    pub(in crate::request) entries: Vec<PairsSaveEntry>,
}

impl Validate for PairsSaveRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.path, "pairs save requires an output path".to_owned())?;
        if self.entries.is_empty() {
            return Err("pairs save requires at least one pair".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct PairsSynthesizeRequest {
    /// Which model writes the pairs: local or brama. The model, revision and
    /// device below belong to the local route only; a brama body omits them.
    /// Neither route is assumed: a body without one is refused.
    #[serde(default)]
    pub(in crate::request) generator: String,
    #[serde(default)]
    pub(in crate::request) generator_model: Option<String>,
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default, rename = "trait")]
    pub(in crate::request) trait_description: String,
    #[serde(default)]
    pub(in crate::request) trait_name: String,
    #[serde(default)]
    pub(in crate::request) opposite: Option<String>,
    /// The dtype the local generator's base weights are mapped at. Ignored by
    /// the `brama` route, which loads no weights.
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: String,
    /// `auto` asks the local generator through the model's own chat template
    /// when it publishes one. Ignored by the `brama` route, which is already
    /// a chat API.
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: String,
    #[serde(default)]
    pub(in crate::request) count: usize,
    #[serde(default)]
    pub(in crate::request) output: String,
    /// Every sampling and filtering number below is the caller's; Ster
    /// assumes none, and a body without one is refused by name.
    pub(in crate::request) retry_multiplier: usize,
    pub(in crate::request) dedupe_bits: u32,
    pub(in crate::request) dedupe_bands: u32,
    pub(in crate::request) refusal_threshold: f32,
    pub(in crate::request) max_new_tokens: usize,
    pub(in crate::request) temperature: f64,
    pub(in crate::request) top_p: f64,
    pub(in crate::request) seed: u64,
}

impl Validate for PairsSynthesizeRequest {
    fn validate(&self) -> Result<(), String> {
        // The local route loads weights and so needs a model; the brama route
        // loads nothing and needs a gateway route instead.
        match self.generator.as_str() {
            "" => return Err("pairs synthesize requires a generator, local or brama".to_owned()),
            "local" => self.model.check("pairs synthesize")?,
            "brama" => require(
                self.generator_model.as_deref().unwrap_or_default(),
                "pairs synthesize with the brama generator requires generatorModel".to_owned(),
            )?,
            _ => return Err("unknown generator; expected local or brama".to_owned()),
        }
        require(
            &self.trait_description,
            "pairs synthesize requires a trait description".to_owned(),
        )?;
        require(
            &self.output,
            "pairs synthesize requires an output path".to_owned(),
        )?;
        if self.count == 0 {
            return Err("pairs synthesize requires a pair count above zero".to_owned());
        }
        Ok(())
    }
}

/// A pair set read from a benchmark export: the same fields as
/// `ster pairs import`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct PairsImportRequest {
    #[serde(default)]
    pub(in crate::request) benchmark: String,
    #[serde(default)]
    pub(in crate::request) source: String,
    #[serde(default)]
    pub(in crate::request) examples: Option<String>,
    #[serde(default)]
    pub(in crate::request) count: Option<usize>,
    pub(in crate::request) seed: u64,
    #[serde(default)]
    pub(in crate::request) trait_name: Option<String>,
    #[serde(default)]
    pub(in crate::request) output: String,
}

impl Validate for PairsImportRequest {
    fn validate(&self) -> Result<(), String> {
        crate::pairs::benchmark::Benchmark::parse(&self.benchmark)
            .map_err(|error| error.to_string())?;
        require(
            &self.source,
            "pairs import requires a source export".to_owned(),
        )?;
        require(
            &self.output,
            "pairs import requires an output path".to_owned(),
        )
    }
}
