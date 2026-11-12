//! The end-of-sequence token a Mistral-format checkpoint leaves to its
//! tokenizer.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::failures::model::tokenizer::{CONFIG_MALFORMED, json_refusal, read_refusal};
use crate::model::keys::{list, mistyped, text, within};

/// The end-of-sequence token a Mistral-format config leaves out: the
/// tokenizer config's `eos_token` looked up among the added tokens. None
/// when there is no tokenizer config or it names no token; a tokenizer file
/// that cannot be read or parsed, or one stating `eos_token`, `added_tokens`
/// or a token's `content` or `id` with the wrong type, is the error, since
/// read as nothing the model was loaded with no end-of-sequence token and
/// never stopped.
pub fn end_of_sequence(tokenizer_config: Option<&Path>, tokenizer: &Path) -> Result<Option<u64>> {
    let read = |path: &Path| -> Result<Value> {
        let bytes =
            std::fs::read(path).map_err(|error| read_refusal("tokenizer file", path, &error))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| json_refusal(&CONFIG_MALFORMED, "tokenizer config", path, &error))
    };
    let Some(tokenizer_config) = tokenizer_config else {
        return Ok(None);
    };
    let config = read(tokenizer_config)?;
    let eos = match config.get("eos_token") {
        None | Some(Value::Null) => None,
        Some(Value::String(eos)) => Some(eos.clone()),
        Some(Value::Object(_)) => text(&config, "eos_token.content", tokenizer_config)?.map(str::to_owned),
        Some(other) => {
            return Err(mistyped(
                "eos_token",
                other,
                "a string or an object with content",
                tokenizer_config,
            ));
        }
    };
    let Some(eos) = eos else {
        return Ok(None);
    };
    let tokens = read(tokenizer)?;
    for (index, token) in list(&tokens, "added_tokens", tokenizer)?.into_iter().flatten().enumerate() {
        let name = format!("added_tokens[{index}]");
        if within::<&str>(token, &name, "content", tokenizer)? == Some(eos.as_str()) {
            return Ok(within::<usize>(token, &name, "id", tokenizer)?.map(|id| id as u64));
        }
    }
    Ok(None)
}
