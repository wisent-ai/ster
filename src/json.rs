//! Writing Ster's own values as JSON: reports, artifacts, sidecars, states.
//!
//! serde_json refuses to write a value for two reasons only: a map key that
//! is not a string, number or boolean (`serde_json::ser::MapKeySerializer`),
//! and a hand-written `impl Serialize` that returns `Error::custom`. Every
//! type src/ writes derives `Serialize` over strings, numbers, booleans,
//! options, vectors, `serde_json::Value` and maps keyed by `String`, `&str`
//! or `usize`, and src/ holds no `impl Serialize` of its own. So no input,
//! checkpoint, config or request can make writing one fail; a type that broke
//! this would fail every run that writes it, which its first test shows.

use serde::Serialize;
use serde_json::Value;

/// What every expect below rests on, quoted from the module comment.
const PLAIN: &str = "src/ holds no `impl Serialize` and no map keyed by anything but a string or an integer";

/// `value` as a JSON value.
pub fn value<T: Serialize + ?Sized>(value: &T) -> Value {
    serde_json::to_value(value).expect(PLAIN)
}

/// `value` as indented JSON text.
pub fn pretty<T: Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string_pretty(value).expect(PLAIN)
}

/// `value` as JSON text on one line.
pub fn compact<T: Serialize + ?Sized>(value: &T) -> String {
    serde_json::to_string(value).expect(PLAIN)
}
