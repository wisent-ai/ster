//! The explicit declaration/output schema shared by native model creators.

use anyhow::Result;
use serde::Deserialize;

use super::{Validate, require};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct ModelCreateRequest {
    #[serde(rename = "from")]
    pub(in crate::request) declaration: String,
    pub(in crate::request) output: String,
}

impl Validate for ModelCreateRequest {
    fn validate(&self) -> Result<()> {
        require(
            &self.declaration,
            "model creation",
            "from",
            "the path of the declaration to create from",
            "--from",
        )?;
        require(
            &self.output,
            "model creation",
            "output",
            "the directory the created artifact is written to",
            "--output",
        )
    }
}
