//! Reading a decoder family's config: the shared key reader
//! ([`crate::model::keys`]) every family reads its keys through, and the
//! per-layer and list readers only the families use. A key stated with the
//! wrong type is refused by the shared reader; only an unstated key takes a
//! default.

pub(in crate::runtime::load) mod activation;
pub(in crate::runtime::load) mod candle;
pub(in crate::runtime::load) mod layers;

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

pub(in crate::runtime::load) use crate::model::keys::{
    describe, first, flag, flag_or, list, mistyped, number, number_or, or, signed, stated, text,
    whole, whole_or, within, within_or,
};

/// A whole number a config states once, or once per layer as HunYuan's
/// lists do. A per-layer list must hold one value throughout, because Ster
/// builds every mixture-of-experts layer alike.
pub(in crate::runtime::load) fn uniform(
    raw: &Value,
    key: &str,
    path: &Path,
) -> Result<Option<usize>> {
    let Some(Value::Array(values)) = raw.get(key) else {
        return whole(raw, key, path);
    };
    let values: Option<Vec<u64>> = values.iter().map(Value::as_u64).collect();
    match values.as_deref() {
        Some([first, rest @ ..]) if rest.iter().all(|value| value == first) => {
            Ok(Some(*first as usize))
        }
        _ => refused!(
            Unimplemented,
            in raw,
            [key],
            "{} lists {key} values that differ between layers or are not whole numbers; Ster builds every mixture-of-experts layer alike",
            path.display()
        ),
    }
}

/// A list of exactly `N` numbers under `key`.
pub(in crate::runtime::load) fn numbers<const N: usize>(
    raw: &Value,
    key: &str,
    path: &Path,
) -> Result<[f64; N]> {
    let stated = list(raw, key, path)?
        .and_then(|list| list.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>());
    match stated.map(<[f64; N]>::try_from) {
        Some(Ok(values)) => Ok(values),
        _ => refused!(
            Malformed,
            in raw,
            [key],
            "{} states {key} as {}, which is not a list of {N} numbers; state {key} as {N} numbers",
            path.display(),
            describe(raw.get(key))
        ),
    }
}
