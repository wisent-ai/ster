//! `--benchmark preference` (IMP-STER-06 DPO, 01 Hugging Face preference
//! datasets, 19 Preferences): rows of `prompt`, `chosen` and `rejected`, the
//! shape TRL's DPO trainer and the Hugging Face preference datasets publish,
//! as JSON Lines or a JSON list. The chosen answer is the positive side and
//! the rejected one the negative, both asked the row's prompt, as every
//! imported pair is written. A side given as chat messages (`[{"role",
//! "content"}]`, the conversational form) is its last message's content, and
//! the messages before it, when the row has no `prompt`, are the question.

use serde_json::Value;

use super::pair;
use crate::ContrastivePair;
use crate::failures::{Rejection, pairs::rows as causes};

/// The text of `value`: itself when a string, else the content of the last
/// message of a chat, with the messages before it.
fn side(value: &Value) -> Option<(String, String)> {
    if let Some(text) = value.as_str().map(str::trim).filter(|text| !text.is_empty()) {
        return Some((String::new(), text.to_owned()));
    }
    let messages = value.as_array()?;
    let (last, before) = messages.split_last()?;
    let answer = last.get("content")?.as_str()?.trim().to_owned();
    let asked = before
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    (!answer.is_empty()).then_some((asked, answer))
}

/// One preference row as the pair `pairs import --benchmark preference` writes, or why it is none.
pub fn pair_of(row: &Value) -> Result<ContrastivePair, Rejection> {
    let read = |field: &'static str, which: &'static str| {
        side(&row[field]).ok_or_else(|| {
            Rejection::new(
                &causes::PREFERENCE_SIDE_MISSING,
                vec![("field", field.to_owned()), ("side", which.to_owned()), ("found", row[field].to_string())],
            )
        })
    };
    let (asked_chosen, chosen) = read("chosen", "positive")?;
    let (_, rejected) = read("rejected", "negative")?;
    let prompt = match &row["prompt"] {
        Value::String(text) if !text.trim().is_empty() => text.trim().to_owned(),
        other => side(other).map_or(asked_chosen, |(asked, said)| if asked.is_empty() { said } else { format!("{asked}\n{said}") }),
    };
    Ok(pair(&prompt, &chosen, &rejected))
}
