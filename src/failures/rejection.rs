//! Rows an import refuses and goes on past: each one's declared cause with
//! the values it names, as the import's report lists it, and the refusal of
//! an import that keeps no row, which is its first refused row's own cause
//! (so the fix a reader meets is that row's) with how every row was refused.
//! `ster decisions import` (`ster.decisions.import.rows`) and `ster pairs
//! import` (`ster.pairs.import.rows`) refuse their rows through it.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use wisent_errors::Cause;

use super::{amend, declared, refuse};

/// Why one row is refused: its cause and the values its sentences name.
pub struct Rejection {
    pub cause: &'static Cause,
    pub values: Vec<(&'static str, String)>,
}

impl Rejection {
    pub fn new(cause: &'static Cause, values: Vec<(&'static str, String)>) -> Self {
        Self { cause, values }
    }

    /// This rejection with `name` set to `value` too: the row and the file,
    /// which the code that reads one row does not know.
    pub fn with(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.values.push((name, value.into()));
        self
    }

    /// The refusal this rejection declares.
    pub fn refusal(&self) -> anyhow::Error {
        let values: Vec<(&str, &str)> =
            self.values.iter().map(|(name, value)| (*name, value.as_str())).collect();
        refuse(self.cause, &values)
    }
}

/// One refused row as a report lists it: the refusal as a reader meets it
/// and its declared cause (`functionality`, `id`, `docs`, `code`, `exit`,
/// `usage`, `retryable`, `remedy`).
#[derive(Debug, Clone, Serialize)]
pub struct Refused {
    pub reason: String,
    pub cause: Value,
}

/// The rows refused so far: the first one's refusal and the count per cause.
#[derive(Debug, Default)]
pub struct Rejections {
    first: Option<anyhow::Error>,
    counts: BTreeMap<&'static str, usize>,
}

impl Rejections {
    /// Count `rejection` and give what the report lists for its row.
    pub fn record(&mut self, rejection: &Rejection) -> Refused {
        let error = rejection.refusal();
        *self.counts.entry(rejection.cause.id).or_insert(0) += 1;
        let refused = Refused {
            reason: error.to_string(),
            cause: declared(&error).map_or(Value::Null, super::Declared::summary),
        };
        self.first.get_or_insert(error);
        refused
    }

    /// The refusal of an import that kept no row: the first refused row's
    /// own refusal, `kept_none` (what keeping no row means) and the count of
    /// rows refused for each cause after it; `None` when no row was refused.
    pub fn refusal(self, kept_none: &str, rows: usize) -> Option<anyhow::Error> {
        let counts: Vec<String> =
            self.counts.iter().map(|(cause, count)| format!("{cause} {count}")).collect();
        let first = self.first?;
        Some(amend(
            first,
            format!("{kept_none}: rows: {rows}; refused by cause: {}", counts.join(", ")),
        ))
    }
}
