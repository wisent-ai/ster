//! The pair-authoring endpoints' requests: importing a set into the
//! workspace, auditing one, saving one, and writing one with a model.

use serde::Deserialize;

use super::defaults::*;
use super::{require, ModelRequest, Validate};


#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::serve) struct WorkspaceImportPairsRequest {
    #[serde(default)]
    pub(in crate::serve) source: String,
    #[serde(default)]
    pub(in crate::serve) name: Option<String>,
}

impl Validate for WorkspaceImportPairsRequest {
    fn validate(&self) -> Result<(), String> {
        require(
            &self.source,
            "workspace import-pairs requires a source path".to_owned(),
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::serve) struct PairsInspectRequest {
    #[serde(default)]
    pub(in crate::serve) pairs: String,
    #[serde(default = "default_dedupe_bits")]
    pub(in crate::serve) dedupe_bits: u32,
    #[serde(default = "default_dedupe_bands")]
    pub(in crate::serve) dedupe_bands: u32,
    #[serde(default = "default_refusal_threshold")]
    pub(in crate::serve) refusal_threshold: f32,
}

impl Validate for PairsInspectRequest {
    fn validate(&self) -> Result<(), String> {
        require(&self.pairs, "pairs inspect requires a pairs file".to_owned())
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::serve) struct PairsSaveEntry {
    #[serde(default)]
    pub(in crate::serve) positive: String,
    #[serde(default)]
    pub(in crate::serve) negative: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::serve) struct PairsSaveRequest {
    #[serde(default)]
    pub(in crate::serve) path: String,
    #[serde(default)]
    pub(in crate::serve) trait_name: String,
    #[serde(default)]
    pub(in crate::serve) entries: Vec<PairsSaveEntry>,
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
pub(in crate::serve) struct PairsSynthesizeRequest {
    /// Which model writes the pairs: local or brama. The model, revision and
    /// device below belong to the local route only; a brama body omits them.
    #[serde(default = "default_generator")]
    pub(in crate::serve) generator: String,
    #[serde(default)]
    pub(in crate::serve) generator_model: Option<String>,
    #[serde(flatten)]
    pub(in crate::serve) model: ModelRequest,
    #[serde(default, rename = "trait")]
    pub(in crate::serve) trait_description: String,
    #[serde(default)]
    pub(in crate::serve) trait_name: String,
    #[serde(default)]
    pub(in crate::serve) opposite: Option<String>,
    /// The dtype the local generator's base weights are mapped at. Ignored by
    /// the `brama` route, which loads no weights.
    #[serde(default = "default_precision")]
    pub(in crate::serve) precision: String,
    /// `auto` asks the local generator through the model's own chat template
    /// when it publishes one. Ignored by the `brama` route, which is already
    /// a chat API.
    #[serde(default = "default_chat_template")]
    pub(in crate::serve) chat_template: String,
    #[serde(default)]
    pub(in crate::serve) count: usize,
    #[serde(default)]
    pub(in crate::serve) output: String,
    #[serde(default = "default_retry_multiplier")]
    pub(in crate::serve) retry_multiplier: usize,
    #[serde(default = "default_dedupe_bits")]
    pub(in crate::serve) dedupe_bits: u32,
    #[serde(default = "default_dedupe_bands")]
    pub(in crate::serve) dedupe_bands: u32,
    #[serde(default = "default_refusal_threshold")]
    pub(in crate::serve) refusal_threshold: f32,
    #[serde(default = "default_synthesis_max_new_tokens")]
    pub(in crate::serve) max_new_tokens: usize,
    #[serde(default = "default_synthesis_temperature")]
    pub(in crate::serve) temperature: f64,
    #[serde(default = "default_top_p")]
    pub(in crate::serve) top_p: f64,
    #[serde(default = "default_seed")]
    pub(in crate::serve) seed: u64,
}

impl Validate for PairsSynthesizeRequest {
    fn validate(&self) -> Result<(), String> {
        // The local route loads weights and so needs a model; the brama route
        // loads nothing and needs a gateway route instead.
        match self.generator.as_str() {
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
        require(&self.output, "pairs synthesize requires an output path".to_owned())?;
        if self.count == 0 {
            return Err("pairs synthesize requires a pair count above zero".to_owned());
        }
        Ok(())
    }
}
