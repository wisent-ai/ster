//! Every way a Ster functionality can fail, declared beside the code that
//! raises it ([`wisent_errors::Operation`]).
//!
//! A functionality is one failure point (`ster.workspace.import`), with the
//! entries a person or a program starts it through and the functionalities it
//! uses. Each cause names the step that failed, why, the state the refusal
//! shows, what fixes it, its code and exit status and the real test that
//! provokes it. The raising code refuses through [`refuse`], so the sentence a
//! reader meets is the declared one with the actual names filled in, and
//! `ster errors list|show|check`, `ster request errors/list|show` and
//! the area pages under docs/guide/errors/ all print the same declaration.

use std::fmt;

use serde_json::{Value, json};
use wisent_errors::{Cause, Operation};

/// The page under docs/guide/errors/ an area's functionalities are listed
/// on, one anchor per cause: `page!("workspace")`. One page per area keeps
/// each page a size a reader can hold as the catalogue grows.
macro_rules! page {
    ($area:literal) => {
        concat!(
            "https://github.com/wisent-ai/ster/blob/main/docs/guide/errors/",
            $area,
            ".md"
        )
    };
}

mod catalogue;
pub mod check;
pub mod pairs;
pub mod workspace;

pub use check::check;

/// The status a failure that is not a usage refusal and not retryable exits
/// with, as `ster` has always exited.
pub const FAILURE_EXIT: i32 = 1;

/// The page that says how a refusal reads, which status it exits with, and
/// which area page lists each functionality's causes.
pub const DOCS: &str = "https://github.com/wisent-ai/ster/blob/main/docs/guide/errors.md";

/// Every functionality that declares how it fails.
pub fn operations() -> Vec<&'static Operation> {
    [
        catalogue::OPERATIONS,
        pairs::OPERATIONS,
        workspace::OPERATIONS,
    ]
    .concat()
}

/// A refusal a declared cause raised, with the values its sentences name,
/// and the error it declares when one was raised underneath it. That error
/// is kept for callers that tell it by its type; its words are already in
/// the sentence, so it is not a `source` the chain would print again.
#[derive(Debug)]
pub struct Declared {
    pub cause: &'static Cause,
    sentence: String,
    fix: String,
    underneath: Option<anyhow::Error>,
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

    /// What a machine reads beside the sentence: the cause's id, code, exit
    /// status, whether retrying unchanged can help, and the fix.
    pub fn summary(&self) -> Value {
        json!({
            "id": self.cause.id,
            "code": self.cause.code.as_str(),
            "exit": self.exit_code(),
            "usage": self.cause.usage,
            "retryable": self.cause.code.retryable(),
            "remedy": self.fix,
        })
    }
}

/// The refusal `cause` declares, its `{name}` placeholders filled from
/// `values`.
pub fn refuse(cause: &'static Cause, values: &[(&str, &str)]) -> anyhow::Error {
    raise(cause, values, None)
}

/// The refusal `cause` declares for `underneath`, which it keeps, so a
/// caller that tells an error by its type still finds it.
pub fn refuse_from(
    cause: &'static Cause,
    values: &[(&str, &str)],
    underneath: anyhow::Error,
) -> anyhow::Error {
    raise(cause, values, Some(underneath))
}

fn raise(
    cause: &'static Cause,
    values: &[(&str, &str)],
    underneath: Option<anyhow::Error>,
) -> anyhow::Error {
    anyhow::Error::new(Declared {
        cause,
        sentence: cause.sentence(values),
        fix: cause.fix(values),
        underneath,
    })
}

/// The declared cause an error carries, wherever in its chain it was raised.
pub fn declared(error: &anyhow::Error) -> Option<&Declared> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<Declared>())
}

/// The refusal a `check` answer with findings ends in, once the answer that
/// lists them is printed; `None` when it found nothing.
pub fn findings(answer: &Value) -> Option<anyhow::Error> {
    let count = answer["findings"].as_array().map_or(0, Vec::len);
    (count > 0).then(|| refuse(&catalogue::FINDINGS, &[("count", &count.to_string())]))
}

/// The status a command that ends in `error` exits with: the declared
/// cause's, or [`FAILURE_EXIT`] for an error no cause declares yet.
pub fn exit_code(error: &anyhow::Error) -> i32 {
    declared(error).map_or(FAILURE_EXIT, Declared::exit_code)
}

/// The functionality `id` names, or the refusal listing the declared ones.
fn named(id: &str) -> anyhow::Result<&'static Operation> {
    let operations = operations();
    match operations.iter().find(|operation| operation.id == id) {
        Some(operation) => Ok(*operation),
        None => {
            let declared = operations
                .iter()
                .map(|operation| operation.id)
                .collect::<Vec<_>>()
                .join(", ");
            Err(refuse(
                &catalogue::FUNCTIONALITY_UNKNOWN,
                &[("id", id), ("declared", &declared)],
            ))
        }
    }
}

fn document(operation: &Operation) -> anyhow::Result<Value> {
    Ok(serde_json::from_str(&operation.to_json(FAILURE_EXIT))?)
}

/// `ster errors list`: every functionality with its entries, what it uses
/// and the causes it declares.
pub fn list() -> anyhow::Result<Value> {
    let operations = operations()
        .into_iter()
        .map(document)
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(json!({"docs": DOCS, "functionalities": operations}))
}

/// The documentation sections of the functionalities listed on `page` (a
/// path such as docs/guide/errors/workspace.md), or of every functionality
/// without one, in declaration order: what each area page holds below its
/// generated-section marker.
pub fn markdown(page: Option<&str>) -> anyhow::Result<String> {
    let all = operations();
    let listed: Vec<&Operation> = all
        .iter()
        .copied()
        .filter(|operation| page.is_none_or(|page| operation.docs.ends_with(&format!("/{page}"))))
        .collect();
    if listed.is_empty() {
        if let Some(page) = page {
            let mut pages: Vec<&str> = all
                .iter()
                .filter_map(|operation| operation.docs.strip_prefix(PAGES))
                .collect();
            pages.dedup();
            return Err(refuse(
                &catalogue::PAGE_UNKNOWN,
                &[("page", page), ("pages", &pages.join(", "))],
            ));
        }
    }
    Ok(listed
        .iter()
        .map(|operation| operation.markdown(FAILURE_EXIT))
        .collect::<Vec<_>>()
        .join("\n"))
}

/// What every area page's address starts with: the repository's blob root.
const PAGES: &str = "https://github.com/wisent-ai/ster/blob/main/";

/// `ster errors show <functionality>`: its own causes and every cause it
/// meets through the functionalities it uses.
pub fn show(id: &str) -> anyhow::Result<Value> {
    let operation = named(id)?;
    let met: Vec<Value> = operation
        .met()
        .into_iter()
        .map(|(declared_by, cause)| {
            json!({
                "declared_by": declared_by,
                "id": cause.id,
                "code": cause.code.as_str(),
                "exit": cause.exit_code(FAILURE_EXIT),
                "failed": cause.failed,
                "cause": cause.cause,
                "observed": cause.observed,
                "remedy": cause.remedy,
                "test": cause.test,
            })
        })
        .collect();
    Ok(json!({"functionality": document(operation)?, "met": met}))
}
