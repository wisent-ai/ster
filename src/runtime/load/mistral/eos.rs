//! The end-of-sequence token a Mistral-format checkpoint leaves to its
//! tokenizer.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

/// The end-of-sequence token a Mistral-format config leaves out: the
/// tokenizer config's `eos_token` looked up among the added tokens. None
/// when there is no tokenizer config or it names no token; a tokenizer file
/// that cannot be read or parsed is the error, since read as nothing the
/// model was loaded with no end-of-sequence token and never stopped.
pub fn end_of_sequence(tokenizer_config: Option<&Path>, tokenizer: &Path) -> Result<Option<u64>> {
    let read = |path: &Path| -> Result<Value> {
        let bytes =
            std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("{} is not JSON", path.display()))
    };
    let Some(tokenizer_config) = tokenizer_config else {
        return Ok(None);
    };
    let text = read(tokenizer_config)?
        .get("eos_token")
        .and_then(|token| match token {
            Value::String(text) => Some(text.clone()),
            Value::Object(object) => object
                .get("content")
                .and_then(Value::as_str)
                .map(str::to_owned),
            _ => None,
        });
    let Some(text) = text else {
        return Ok(None);
    };
    Ok(read(tokenizer)?
        .get("added_tokens")
        .and_then(Value::as_array)
        .and_then(|tokens| {
            tokens
                .iter()
                .find(|token| token.get("content").and_then(Value::as_str) == Some(text.as_str()))
        })
        .and_then(|token| token.get("id"))
        .and_then(Value::as_u64))
}
