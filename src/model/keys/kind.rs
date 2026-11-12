//! The types a config key is read as, each with the name a refusal of a
//! key stated with another type gives it.

use serde_json::{Map, Value};

/// A type a config key is read as, and how a refusal names it.
pub(crate) trait Kind<'a>: Sized {
    /// The type as a refusal names it: "a whole number".
    const KIND: &'static str;
    /// The value as this type, or `None` when it is another type.
    fn read(value: &'a Value) -> Option<Self>;
}

impl Kind<'_> for usize {
    const KIND: &'static str = "a whole number";
    fn read(value: &Value) -> Option<Self> {
        value.as_u64().map(|whole| whole as usize)
    }
}

impl Kind<'_> for u32 {
    const KIND: &'static str = "a token id, a whole number below 4294967296";
    fn read(value: &Value) -> Option<Self> {
        match value.as_u64() {
            Some(whole) if whole <= u64::from(u32::MAX) => Some(whole as u32),
            _ => None,
        }
    }
}

impl Kind<'_> for i64 {
    const KIND: &'static str = "an integer";
    fn read(value: &Value) -> Option<Self> {
        value.as_i64()
    }
}

impl Kind<'_> for f64 {
    const KIND: &'static str = "a number";
    fn read(value: &Value) -> Option<Self> {
        value.as_f64()
    }
}

impl Kind<'_> for bool {
    const KIND: &'static str = "true or false";
    fn read(value: &Value) -> Option<Self> {
        value.as_bool()
    }
}

impl<'a> Kind<'a> for &'a str {
    const KIND: &'static str = "a string";
    fn read(value: &'a Value) -> Option<Self> {
        value.as_str()
    }
}

impl<'a> Kind<'a> for &'a [Value] {
    const KIND: &'static str = "a list";
    fn read(value: &'a Value) -> Option<Self> {
        value.as_array().map(Vec::as_slice)
    }
}

impl<'a> Kind<'a> for &'a Map<String, Value> {
    const KIND: &'static str = "an object";
    fn read(value: &'a Value) -> Option<Self> {
        value.as_object()
    }
}
