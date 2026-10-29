//! The explicit declaration/output schema shared by native model creators.

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
    fn validate(&self) -> Result<(), String> {
        require(
            &self.declaration,
            "model creation requires from (the declaration path)".into(),
        )?;
        require(
            &self.output,
            "model creation requires an output path".into(),
        )
    }
}
