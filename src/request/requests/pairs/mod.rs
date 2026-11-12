//! The pair-authoring operations' requests: importing a set into the
//! workspace, auditing one, saving one, and writing one with a model.

use std::num::NonZeroUsize;

use anyhow::Result;
use serde::Deserialize;

use ster::{
    ChatChoice, Precision,
    failures::{pairs::synthesize, refuse, request::too_few},
    pairs::{InspectOptions, quality::dedupe::DedupeOptions},
};

use super::defaults::*;
use super::{ModelRequest, Validate, require};

mod sets;

pub(in crate::request) use sets::{PairsImportRequest, PairsMergeRequest};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct WorkspaceImportPairsRequest {
    #[serde(default)]
    pub(in crate::request) source: String,
    #[serde(default)]
    pub(in crate::request) name: Option<String>,
}

impl Validate for WorkspaceImportPairsRequest {
    fn validate(&self) -> Result<()> {
        require(
            &self.source,
            "workspace import",
            "source",
            "the path of the pair set to import",
            "the SOURCE argument of `ster workspace import`",
        )
    }
}

/// `workspace/show`: no fields; the answer is `ster workspace show`'s.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::request) struct WorkspaceShowRequest {}

impl Validate for WorkspaceShowRequest {
    fn validate(&self) -> Result<()> {
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
    fn validate(&self) -> Result<()> {
        require(
            &self.id,
            "workspace pair-set",
            "id",
            "the set's workspace name, as workspace/show lists it",
            "the ID argument of `ster workspace select` and `ster workspace remove`",
        )
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct PairsInspectRequest {
    #[serde(default)]
    pub(in crate::request) pairs: String,
    /// The deduplication and balance settings are the caller's; Ster assumes
    /// none, and a body without one is refused by name.
    pub(in crate::request) dedupe_bits: u32,
    pub(in crate::request) dedupe_bands: u32,
    pub(in crate::request) unbalanced_ratio: f64,
}

impl Validate for PairsInspectRequest {
    fn validate(&self) -> Result<()> {
        require(
            &self.pairs,
            "pairs inspect",
            "pairs",
            "the path of a pair set",
            "--pairs",
        )?;
        InspectOptions {
            dedupe: DedupeOptions::new(self.dedupe_bits, self.dedupe_bands),
            unbalanced_ratio: self.unbalanced_ratio,
        }
        .validate()
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
    fn validate(&self) -> Result<()> {
        require(
            &self.path,
            "pairs save",
            "path",
            "the path the pair set is written to",
            "`ster pairs add`'s --pairs",
        )?;
        if self.entries.is_empty() {
            return Err(too_few(
                "pairs save",
                "entries",
                "pair",
                1,
                0,
                "`ster pairs add`'s --positive and --negative",
            ));
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
    pub(in crate::request) precision: Precision,
    /// `auto` asks the local generator through the model's own chat template
    /// when it publishes one. Ignored by the `brama` route, which is already
    /// a chat API.
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: ChatChoice,
    /// Above zero; serde refuses a zero, and a body without one is refused in `validate`.
    #[serde(default)]
    pub(in crate::request) count: Option<NonZeroUsize>,
    #[serde(default)]
    pub(in crate::request) output: String,
    /// Every sampling and filtering number below is the caller's; Ster
    /// assumes none, and a body without one is refused by name.
    /// Above zero; serde refuses a zero or a missing multiplier by name.
    pub(in crate::request) retry_multiplier: NonZeroUsize,
    pub(in crate::request) dedupe_bits: u32,
    pub(in crate::request) dedupe_bands: u32,
    pub(in crate::request) max_new_tokens: usize,
    pub(in crate::request) temperature: f64,
    pub(in crate::request) top_p: f64,
    pub(in crate::request) seed: u64,
}

impl Validate for PairsSynthesizeRequest {
    fn validate(&self) -> Result<()> {
        let declared = refuse;
        // The local route loads weights and so needs a model; the brama route
        // loads nothing and needs the route Stado's inference alternatives are
        // asked for instead.
        match self.generator.as_str() {
            "local" if self.model.model.trim().is_empty() => {
                return Err(declared(
                    &synthesize::MODEL_MISSING,
                    &[("trait", &self.trait_description)],
                ));
            }
            "brama"
                if self
                    .generator_model
                    .as_deref()
                    .is_none_or(|route| route.trim().is_empty()) =>
            {
                return Err(declared(
                    &synthesize::ROUTE_MISSING,
                    &[("trait", &self.trait_description)],
                ));
            }
            "local" | "brama" => {}
            other => {
                return Err(declared(
                    &synthesize::GENERATOR_UNKNOWN,
                    &[("generator", other)],
                ));
            }
        }
        if self.trait_description.trim().is_empty() {
            return Err(declared(
                &synthesize::TRAIT_BLANK,
                &[("trait", &self.trait_description)],
            ));
        }
        require(
            &self.output,
            "pairs synthesize",
            "output",
            "the path the written pair set goes to",
            "--output",
        )?;
        // Refused here, before a local model loads, rather than inside the run.
        if self.count.is_none() {
            return Err(declared(
                &synthesize::COUNT_MISSING,
                &[("trait", &self.trait_description)],
            ));
        }
        DedupeOptions::new(self.dedupe_bits, self.dedupe_bands).validate()
    }
}
