//! Example sets as other trainers publish them, read where `ExampleSet::load`
//! finds a JSON list or JSON Lines instead of `{"examples": [...]}`. Each row's
//! form is read from the fields its format declares:
//!
//! - OpenAI's fine-tuning JSONL (IMP-STER-03): `messages`, each with a `role`
//!   (`system`, `user`, `assistant`) and its `content`; the legacy rows of
//!   `prompt` and `completion` are an example as they stand.
//! - ShareGPT (04): `conversations`, each turn's `from` (`system`, `human`,
//!   `gpt`) and its `value`.
//! - Alpaca (05): `instruction`, an optional `input` and `output`; the prompt is
//!   the instruction followed, after a blank line, by the input when the row has
//!   one, since an Alpaca row states both as the request.
//!
//! An example is one prompt and its completion, under an optional system turn, so
//! a conversation is one when it is an optional system turn, one turn of the user
//! and one of the assistant, in that order; a conversation of more turns, and a
//! row of none of these forms, is refused by its number.

use anyhow::Result;
use serde_json::Value;

use super::examples::Example;
use crate::failures::{
    refuse,
    tune::examples::{CONVERSATION_UNFIT, ROW_UNKNOWN},
};

/// A turn's speaker as the format names it, read as the three an example has.
fn speaker(name: &str) -> Option<&'static str> {
    match name {
        "system" => Some("system"),
        "user" | "human" => Some("user"),
        "assistant" | "gpt" => Some("assistant"),
        _ => None,
    }
}

fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

/// The example a conversation of `(speaker, text)` turns is.
fn conversation(turns: &[Value], role: &str, content: &str, label: &str, index: usize) -> Result<Example> {
    let said: Vec<(Option<&'static str>, Option<String>)> = turns
        .iter()
        .map(|turn| (turn[role].as_str().and_then(speaker), text(&turn[content])))
        .collect();
    let shape: Vec<&str> = said.iter().map(|(who, _)| who.unwrap_or("unknown")).collect();
    let refused = || {
        refuse(
            &CONVERSATION_UNFIT,
            &[("label", label), ("index", &index.to_string()), ("turns", &shape.join(", "))],
        )
    };
    let (system, rest) = match said.as_slice() {
        [(Some("system"), Some(system)), rest @ ..] => (Some(system.clone()), rest),
        rest => (None, rest),
    };
    match rest {
        [(Some("user"), Some(prompt)), (Some("assistant"), Some(completion))] => {
            Ok(Example { system, prompt: prompt.clone(), completion: completion.clone() })
        }
        _ => Err(refused()),
    }
}

/// The example row `index` of the set `label` is.
fn example(row: &Value, label: &str, index: usize) -> Result<Example> {
    if let Some(turns) = row["messages"].as_array() {
        return conversation(turns, "role", "content", label, index);
    }
    if let Some(turns) = row["conversations"].as_array() {
        return conversation(turns, "from", "value", label, index);
    }
    if let (Some(instruction), Some(output)) = (text(&row["instruction"]), text(&row["output"])) {
        let prompt = match text(&row["input"]).filter(|input| !input.trim().is_empty()) {
            Some(input) => format!("{instruction}\n\n{input}"),
            None => instruction,
        };
        return Ok(Example { system: None, prompt, completion: output });
    }
    if let (Some(prompt), Some(completion)) = (text(&row["prompt"]), text(&row["completion"])) {
        return Ok(Example { system: text(&row["system"]), prompt, completion });
    }
    let fields = row.as_object().map_or_else(|| row.to_string(), |object| object.keys().cloned().collect::<Vec<_>>().join(", "));
    Err(refuse(&ROW_UNKNOWN, &[("label", label), ("index", &index.to_string()), ("fields", &fields)]))
}

/// The rows of JSON Lines (wisent-formats' one reader), read when `bytes` is not
/// one JSON value; none when a line is not JSON either.
pub(super) fn lines(bytes: &[u8]) -> Option<Vec<Value>> {
    let rows = wisent_formats::jsonl::read(bytes).ok()?;
    (!rows.is_empty()).then(|| rows.into_iter().map(|row| row.value).collect())
}

/// The examples of imported `rows`, every row one example.
pub(super) fn examples(rows: &[Value], label: &str) -> Result<Vec<Example>> {
    rows.iter().enumerate().map(|(index, row)| example(row, label, index)).collect()
}
