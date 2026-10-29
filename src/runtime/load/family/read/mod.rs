//! Reading one config key: the scalar readers every family shares. A key
//! read with [`stated_whole`] or [`uniform`] is refused when the config
//! states it with the wrong type; [`whole`], [`number`] and [`text`] read a
//! mistyped key as unstated.

pub(in crate::runtime::load::family) mod activation;
pub(in crate::runtime::load::family) mod candle;
pub(in crate::runtime::load::family) mod layers;

use std::path::Path;

use anyhow::{Result, bail};
use serde_json::Value;

/// What `key` states, read by `read`; `None` when the key is absent or
/// `null` (Python's `None`, and the non-finite floats `config::read` turns
/// into `null`), and a refusal naming the key, the value and the type
/// `kind` when it is present but `read` cannot read it.
fn stated<'a, T>(
    raw: &'a Value,
    key: &str,
    kind: &str,
    read: impl Fn(&'a Value) -> Option<T>,
    path: &Path,
) -> Result<Option<T>> {
    match raw.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => match read(value) {
            Some(read) => Ok(Some(read)),
            None => bail!(
                "{} states {key} as {value}, which is not {kind}; state {key} as {kind}",
                path.display()
            ),
        },
    }
}

/// The whole number `key` states, if it states one.
pub(in crate::runtime::load::family) fn stated_whole(
    raw: &Value,
    key: &str,
    path: &Path,
) -> Result<Option<usize>> {
    stated(
        raw,
        key,
        "a whole number",
        |value| value.as_u64().map(|whole| whole as usize),
        path,
    )
}

/// A whole number a config states once, or once per layer as HunYuan's
/// lists do. A per-layer list must hold one value throughout, because Ster
/// builds every mixture-of-experts layer alike.
pub(in crate::runtime::load::family) fn uniform(
    raw: &Value,
    key: &str,
    path: &Path,
) -> Result<Option<usize>> {
    let Some(Value::Array(values)) = raw.get(key) else {
        return stated_whole(raw, key, path);
    };
    let values: Option<Vec<u64>> = values.iter().map(Value::as_u64).collect();
    match values.as_deref() {
        Some([first, rest @ ..]) if rest.iter().all(|value| value == first) => {
            Ok(Some(*first as usize))
        }
        _ => bail!(
            "{} lists {key} values that differ between layers or are not whole numbers; Ster builds every mixture-of-experts layer alike",
            path.display()
        ),
    }
}

/// A list of exactly `N` numbers under `key`.
pub(in crate::runtime::load::family) fn numbers<const N: usize>(
    raw: &Value,
    key: &str,
    path: &Path,
) -> Result<[f64; N]> {
    let stated = raw
        .get(key)
        .and_then(Value::as_array)
        .and_then(|list| list.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>());
    match stated.map(<[f64; N]>::try_from) {
        Some(Ok(values)) => Ok(values),
        _ => bail!(
            "{} states {key} as {}, which is not a list of {N} numbers; state {key} as {N} numbers",
            path.display(),
            describe(raw.get(key))
        ),
    }
}

/// A key's value as a refusal shows it, `absent` when the config leaves the
/// key out.
pub(in crate::runtime::load::family) fn describe(value: Option<&Value>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "absent".to_owned(),
    }
}

/// Whether `key` is stated `true`; absent, `null` and `false` all read as
/// off, the value every config class Ster reads gives the switches it reads
/// through this.
pub(in crate::runtime::load::family) fn flag(raw: &Value, key: &str) -> bool {
    raw.get(key).and_then(Value::as_bool) == Some(true)
}

/// The whole number `key` states, if it states one; a key that is not a
/// whole number reads as unstated. Use [`stated_whole`] where a mistyped
/// value must be refused.
pub(in crate::runtime::load::family) fn whole(raw: &Value, key: &str) -> Option<usize> {
    raw.get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
}

/// The number `key` states, if it states one.
pub(in crate::runtime::load::family) fn number(raw: &Value, key: &str) -> Option<f64> {
    raw.get(key).and_then(Value::as_f64)
}

/// The string `key` states, if it states one.
pub(in crate::runtime::load::family) fn text<'a>(raw: &'a Value, key: &str) -> Option<&'a str> {
    raw.get(key).and_then(Value::as_str)
}
