//! The failure catalogue's requests: the same answers `ster errors list`
//! and `ster errors show` print.

use anyhow::Result;
use serde::Deserialize;

use super::{Validate, require};

/// `errors/list`: no fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::request) struct ErrorsListRequest {}

impl Validate for ErrorsListRequest {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// `errors/show`: one functionality by its id, as `errors/list` names it.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(in crate::request) struct ErrorsShowRequest {
    #[serde(default)]
    pub(in crate::request) id: String,
}

impl Validate for ErrorsShowRequest {
    fn validate(&self) -> Result<()> {
        require(
            &self.id,
            "errors/show",
            "id",
            "the functionality's id, as errors/list names it",
            "the FUNCTIONALITY argument of `ster errors show`",
        )
    }
}
