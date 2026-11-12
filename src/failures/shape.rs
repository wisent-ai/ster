//! What a refusal shows of a value it found where another was expected: the
//! kind of a JSON value with enough of it to recognise, and an excerpt of a
//! line it could not read.

use serde_json::Value;

/// How much of a value or a line a refusal quotes: the length Ster's other
/// refusals of one stated value quote (src/model/keys/mod.rs, a config key's
/// value), enough to recognise a row without printing a whole document.
const QUOTED: usize = 120;

/// `value` as a refusal names what was found: `absent`, `null`, `true`, `the
/// number -1`, `text "B"`, `a list of 2`, `an object with keys a, b`.
pub fn shape(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return "absent".to_owned();
    };
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => format!("the number {number}"),
        Value::String(text) => format!("text {}", excerpt(&format!("{text:?}"))),
        Value::Array(items) => format!("a list of {}", items.len()),
        Value::Object(fields) if fields.is_empty() => "an object with no keys".to_owned(),
        Value::Object(fields) => {
            let keys: Vec<&str> = fields.keys().map(String::as_str).collect();
            format!("an object with keys {}", excerpt(&keys.join(", ")))
        }
    }
}

/// `text` cut to the length a refusal quotes.
pub fn excerpt(text: &str) -> String {
    wisent_errors::trim_detail(text, QUOTED)
}
