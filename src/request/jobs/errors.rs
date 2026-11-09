//! Reading the failure catalogue: the same documents `ster errors list` and
//! `ster errors show` print.

use anyhow::Result;
use serde_json::Value;

use super::super::requests::{ErrorsListRequest, ErrorsShowRequest};

/// The same answer as `ster errors list`.
pub(in crate::request) fn errors_list_job(_request: ErrorsListRequest) -> Result<Value> {
    ster::failures::list()
}

/// The same answer as `ster errors show`.
pub(in crate::request) fn errors_show_job(request: ErrorsShowRequest) -> Result<Value> {
    ster::failures::show(&request.id)
}
