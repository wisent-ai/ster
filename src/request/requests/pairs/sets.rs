//! The requests that write a pair set from others: one read from a
//! benchmark export, and several merged into one.

use anyhow::Result;
use serde::Deserialize;

use super::super::{Validate, require};

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
    pub(in crate::request) question: Option<String>,
    #[serde(default)]
    pub(in crate::request) choices: Option<String>,
    #[serde(default)]
    pub(in crate::request) answer: Option<String>,
    #[serde(default)]
    pub(in crate::request) answer_form: Option<String>,
    #[serde(default)]
    pub(in crate::request) labels: Option<String>,
    #[serde(default)]
    pub(in crate::request) count: Option<usize>,
    pub(in crate::request) seed: u64,
    #[serde(default)]
    pub(in crate::request) trait_name: Option<String>,
    #[serde(default)]
    pub(in crate::request) output: String,
}

impl Validate for PairsImportRequest {
    fn validate(&self) -> Result<()> {
        let benchmark = ster::pairs::benchmark::Benchmark::parse(&self.benchmark)?;
        ster::pairs::benchmark::ChoiceFields::from_parts(
            benchmark,
            self.question.clone(),
            self.choices.clone(),
            self.answer.clone(),
            self.answer_form.clone(),
            self.labels.clone(),
        )?;
        require(
            &self.source,
            "pairs import",
            "source",
            "the path of the benchmark export",
            "--source",
        )?;
        require(
            &self.output,
            "pairs import",
            "output",
            "the path the pair set is written to",
            "--output",
        )
    }
}

/// Several pair sets written as one: the same fields as `ster pairs merge`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct PairsMergeRequest {
    /// The pair sets to merge, two or more, in order.
    pub(in crate::request) sources: Vec<String>,
    pub(in crate::request) trait_name: String,
    #[serde(default)]
    pub(in crate::request) output: String,
}

impl Validate for PairsMergeRequest {
    fn validate(&self) -> Result<()> {
        for (index, source) in self.sources.iter().enumerate() {
            require(
                source,
                "pairs merge",
                &format!("sources[{index}]"),
                "the path of a pair set to merge",
                "each --pairs",
            )?;
        }
        require(
            &self.output,
            "pairs merge",
            "output",
            "the path the merged set is written to",
            "--output",
        )
    }
}
