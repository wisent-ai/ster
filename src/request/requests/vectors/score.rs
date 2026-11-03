//! `ster request vector/score`: the request of `ster vector score`, every flag
//! a camelCase field.

use serde::Deserialize;

use super::super::{ModelRequest, Validate, defaults::*, require};
use ster::{ChatChoice, Precision};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct ScoreRequest {
    #[serde(flatten)]
    pub(in crate::request) model: ModelRequest,
    #[serde(default)]
    pub(in crate::request) vector: String,
    /// The layer every token is read at; required.
    pub(in crate::request) layer: usize,
    /// The text to score.
    #[serde(default)]
    pub(in crate::request) text: String,
    #[serde(default = "default_chat_template")]
    pub(in crate::request) chat_template: ChatChoice,
    #[serde(default = "default_precision")]
    pub(in crate::request) precision: Precision,
}

impl Validate for ScoreRequest {
    fn validate(&self) -> Result<(), String> {
        self.model.check("vector/score")?;
        require(&self.vector, "vector/score requires a steering artifact".to_owned())?;
        require(&self.text, "vector/score requires the text to score".to_owned())
    }
}
