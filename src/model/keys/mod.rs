//! Reading one key of a checkpoint's config: the one reader every decoder
//! family's reading, the Llama keys Candle needs, the rotation, and the
//! Mistral and multimodal conversions read a config through ([`stated`]).
//!
//! A key the config leaves out, or states as `null` (Python's `None`, and
//! the non-finite floats the config reader turns into `null`), is unstated,
//! and only an unstated key takes the default an `_or` reader names. A key
//! stated with another type is refused as `ster.model.config.key-mistyped`,
//! naming the key, the value the config states and the type the reading
//! expects, instead of being read as unstated. A dotted key
//! (`attn_config.alibi`) is read through the objects it names.

mod kind;

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::failures::{model::config as causes, refuse};

pub(crate) use kind::Kind;

/// The value `key` states in `raw`, read as `T`: `None` when the key is
/// unstated, the refusal [`mistyped`] when it is stated as another type.
pub(crate) fn stated<'a, T: Kind<'a>>(raw: &'a Value, key: &str, path: &Path) -> Result<Option<T>> {
    typed(at(raw, key), key, path)
}

/// What `key` states in `section`, an object the config nests under
/// `name` (`rope_scaling`, `block_configs[3].ffn`), read as `T`; a
/// refusal names the key as `name.key`.
pub(crate) fn within<'a, T: Kind<'a>>(
    section: &'a Value,
    name: &str,
    key: &str,
    path: &Path,
) -> Result<Option<T>> {
    typed(section.get(key), &format!("{name}.{key}"), path)
}

/// [`within`], or `default` when the section leaves `key` unstated.
pub(crate) fn within_or<'a, T: Kind<'a>>(
    section: &'a Value,
    name: &str,
    key: &str,
    default: T,
    path: &Path,
) -> Result<T> {
    Ok(match within(section, name, key, path)? {
        Some(value) => value,
        None => default,
    })
}

fn typed<'a, T: Kind<'a>>(value: Option<&'a Value>, shown: &str, path: &Path) -> Result<Option<T>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => match T::read(value) {
            Some(read) => Ok(Some(read)),
            None => Err(mistyped(shown, value, T::KIND, path)),
        },
    }
}

/// What the first of `keys` the config states holds, read as `T`: the
/// spellings one setting goes by (`sliding_window_size`, then
/// `sliding_window`). A spelling stated with the wrong type is refused, not
/// passed over.
pub(crate) fn first<'a, T: Kind<'a>>(
    raw: &'a Value,
    keys: &[&str],
    path: &Path,
) -> Result<Option<T>> {
    for key in keys {
        if let Some(value) = stated(raw, key, path)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// What `key` states, read as `T`, or `default` when it is unstated.
pub(crate) fn or<'a, T: Kind<'a>>(raw: &'a Value, key: &str, default: T, path: &Path) -> Result<T> {
    Ok(match stated(raw, key, path)? {
        Some(value) => value,
        None => default,
    })
}

/// The value at `key`, a dotted key read through the objects it names.
fn at<'a>(raw: &'a Value, key: &str) -> Option<&'a Value> {
    key.split('.').try_fold(raw, |object, part| object.get(part))
}

/// The refusal of `key`, stated in the config at `path` as `value`, which
/// is not `kind`.
pub(crate) fn mistyped(key: &str, value: &Value, kind: &str, path: &Path) -> anyhow::Error {
    refuse(
        &causes::KEY_MISTYPED,
        &[
            ("path", &path.display().to_string()),
            ("key", key),
            ("value", &value.to_string().trim()),
            ("kind", kind),
        ],
    )
}

/// The whole number `key` states, if it states one.
pub(crate) fn whole(raw: &Value, key: &str, path: &Path) -> Result<Option<usize>> {
    stated(raw, key, path)
}

/// The whole number `key` states, or `default` when it is unstated.
pub(crate) fn whole_or(raw: &Value, key: &str, default: usize, path: &Path) -> Result<usize> {
    or(raw, key, default, path)
}

/// The integer `key` states, negative ones included, if it states one.
pub(crate) fn signed(raw: &Value, key: &str, path: &Path) -> Result<Option<i64>> {
    stated(raw, key, path)
}

/// The number `key` states, if it states one.
pub(crate) fn number(raw: &Value, key: &str, path: &Path) -> Result<Option<f64>> {
    stated(raw, key, path)
}

/// The number `key` states, or `default` when it is unstated.
pub(crate) fn number_or(raw: &Value, key: &str, default: f64, path: &Path) -> Result<f64> {
    or(raw, key, default, path)
}

/// The string `key` states, if it states one.
pub(crate) fn text<'a>(raw: &'a Value, key: &str, path: &Path) -> Result<Option<&'a str>> {
    stated(raw, key, path)
}

/// The switch `key` states; unstated reads as off, the value every config
/// class Ster reads gives the switches it reads through this.
pub(crate) fn flag(raw: &Value, key: &str, path: &Path) -> Result<bool> {
    or(raw, key, false, path)
}

/// The switch `key` states, or `default` when it is unstated.
pub(crate) fn flag_or(raw: &Value, key: &str, default: bool, path: &Path) -> Result<bool> {
    or(raw, key, default, path)
}

/// The list `key` states, if it states one.
pub(crate) fn list<'a>(raw: &'a Value, key: &str, path: &Path) -> Result<Option<&'a [Value]>> {
    stated(raw, key, path)
}

/// A key's value as a refusal shows it, `absent` when the config leaves the
/// key out.
pub(crate) fn describe(value: Option<&Value>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "absent".to_owned(),
    }
}
