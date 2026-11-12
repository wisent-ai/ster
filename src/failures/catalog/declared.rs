//! The refusal a declared cause raised, and what a machine reads of it: the
//! functionality that declares the cause, the cause's line on that
//! functionality's page, its id, code, exit status and fix.

use std::fmt;

use serde_json::{Value, json};
use wisent_errors::{Cause, Operation};

use super::{DOCS, FAILURE_EXIT, operations};

/// A refusal a declared cause raised, with the values its sentences name,
/// and the error it declares when one was raised underneath it. That error
/// is kept for callers that tell it by its type; its words are already in
/// the sentence, so it is not a `source` the chain would print again.
#[derive(Debug)]
pub struct Declared {
    pub cause: &'static Cause,
    pub(super) sentence: String,
    pub(super) fix: String,
    pub(super) underneath: Option<anyhow::Error>,
}

impl fmt::Display for Declared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}; fix: {}", self.sentence, self.fix)
    }
}

impl std::error::Error for Declared {}

impl Declared {
    /// The exit status the declaration gives this refusal.
    pub fn exit_code(&self) -> i32 {
        self.cause.exit_code(FAILURE_EXIT)
    }

    /// The error this refusal declares, when it declares one.
    pub fn underneath(&self) -> Option<&anyhow::Error> {
        self.underneath.as_ref()
    }

    /// The functionality whose catalogue declares this cause, Ster's own or
    /// a Wisent library's Ster lists (`wisent-errors.telemetry.record`);
    /// `None` for a cause no listed functionality declares.
    pub fn functionality(&self) -> Option<&'static Operation> {
        operations().into_iter().find(|operation| {
            operation.causes.iter().any(|cause| std::ptr::eq(*cause, self.cause))
        })
    }

    /// What a machine reads beside the sentence: `functionality`, the id
    /// `ster errors show` takes for the declaration (`null` when no listed
    /// functionality declares the cause), `docs`, the cause's line on that
    /// functionality's page (`<page>#<functionality>-<cause>`, or the page
    /// on how Ster fails when there is none), the cause's `id`, `code`,
    /// `exit` status, `usage`, whether retrying unchanged can help, and the
    /// fix with the actual values filled in.
    pub fn summary(&self) -> Value {
        let functionality = self.functionality();
        json!({
            "functionality": functionality.map(|operation| operation.id),
            "id": self.cause.id,
            "docs": functionality.map_or_else(
                || DOCS.to_owned(),
                |operation| format!("{}#{}", operation.docs, self.cause.anchor(operation)),
            ),
            "code": self.cause.code.as_str(),
            "exit": self.exit_code(),
            "usage": self.cause.usage,
            "retryable": self.cause.code.retryable(),
            "remedy": self.fix,
        })
    }

    /// The envelope this refusal reports: the failure point of the
    /// functionality that declares the cause, its code, and the sentence and
    /// fix with the actual names filled in.
    pub fn failure(&self) -> wisent_errors::Failure {
        let point = self.functionality().map_or("ster", |operation| operation.id);
        wisent_errors::Failure::or_fallback(point, self.cause.code, "ster")
            .detail(self.sentence.as_str())
            .remedy(self.fix.as_str())
    }
}
