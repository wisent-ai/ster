//! What a refusal shows of a value it found where another was expected: the
//! kind of a JSON value with enough of it to recognise, and an excerpt of a
//! line it could not read.

use serde_json::Value;

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

/// `text` as a refusal quotes it: whole, its ends trimmed. A cut at a chosen length dropped
/// the part past it, which was often the part that said why.
pub fn excerpt(text: &str) -> String {
    text.trim().to_owned()
}
