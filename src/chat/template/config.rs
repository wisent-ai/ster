//! What a chat template reads from the tokenizer config beside it: the
//! template itself, in either shape `transformers` wrote it, and the special
//! tokens it writes.

use anyhow::Result;
use serde_json::Value;

use crate::failures::{chat as causes, refuse};

/// The `chat_template` field of the tokenizer config `path`, in either
/// shape `transformers` ever wrote it.
///
/// The list form pairs a template with a name, and only the one called
/// `default` is the conversation format; the others are tool or RAG variants
/// that expect inputs Ster does not have.
pub(super) fn embedded_template(config: &Value, path: &str) -> Result<Option<String>> {
    match config.get("chat_template") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(source)) => Ok(Some(source.clone())),
        Some(Value::Array(entries)) => {
            let named = entries.iter().find(|entry| name(entry).as_deref() == Some("default"));
            let Some(entry) = named.or_else(|| entries.first().filter(|_| entries.len() == 1)) else {
                let names: Vec<String> = entries.iter().map(shown_name).collect();
                return Err(refuse(
                    &causes::FORM_UNCHOSEN,
                    &[("names", &names.join(", ")), ("path", path)],
                ));
            };
            match entry.get("template").and_then(Value::as_str) {
                Some(source) => Ok(Some(source.to_owned())),
                None => Err(refuse(
                    &causes::FORM_EMPTY,
                    &[
                        ("name", &shown_name(entry)),
                        ("entry", &entry.to_string()),
                        ("path", path),
                    ],
                )),
            }
        }
        Some(other) => Err(refuse(
            &causes::FORM_UNKNOWN,
            &[("value", &other.to_string()), ("path", path)],
        )),
    }
}

/// The `name` a listed template entry gives itself.
fn name(entry: &Value) -> Option<String> {
    entry.get("name").and_then(Value::as_str).map(str::to_owned)
}

/// A listed entry's name quoted, or `(unnamed)`.
fn shown_name(entry: &Value) -> String {
    name(entry).map_or("(unnamed)".to_owned(), |name| format!("{name:?}"))
}

/// A special token as its text, from either the bare string or the
/// `AddedToken` object `transformers` also writes.
pub(super) fn token_text(config: &Value, key: &str) -> Option<String> {
    match config.get(key)? {
        Value::String(text) => Some(text.clone()),
        Value::Object(object) => object.get("content")?.as_str().map(str::to_owned),
        _ => None,
    }
}
