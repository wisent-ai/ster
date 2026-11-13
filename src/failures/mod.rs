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

use serde_json::Value;
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

#[path = "models/candle.rs"]
mod candle;
#[path = "catalog/catalogue.rs"]
mod catalogue;
#[path = "asks/chat.rs"]
pub mod chat;
#[path = "catalog/check.rs"]
pub mod check;
pub mod decide;
#[path = "catalog/declared.rs"]
mod declared;
pub mod file;
pub mod fixture;
#[path = "asks/inference.rs"]
pub mod inference;
pub mod lora;
pub mod model;
pub mod onboarding;
pub mod pairs;
#[path = "asks/rejection.rs"]
mod rejection;
#[path = "asks/request.rs"]
pub mod request;
pub mod runtime;
#[path = "models/shape.rs"]
mod shape;
#[path = "catalog/star.rs"]
pub mod star;
pub mod sources;
pub mod steering;
pub mod telemetry;
#[path = "models/tensor.rs"]
pub mod tensor;
#[path = "models/tokenizer.rs"]
pub mod tokenizer;
pub mod tune;
pub mod workflow;
pub mod workspace;

pub use candle::{lift, lifted, refuse_candle, root as candle_root, shapes_differ};
pub use catalogue::{list, markdown, show};
pub use check::check;
pub use declared::Declared;
pub use rejection::{Refused, Rejection, Rejections};
pub use shape::{excerpt, shape};

/// The status a failure that is not a usage refusal and not retryable exits
/// with, as `ster` has always exited.
pub const FAILURE_EXIT: i32 = 1;

/// The page that says how a refusal reads, which status it exits with, and
/// which area page lists each functionality's causes.
pub const DOCS: &str = "https://github.com/wisent-ai/ster/blob/main/docs/guide/errors.md";

/// Every functionality that declares how it fails.
pub fn operations() -> Vec<&'static Operation> {
    let mut all = [
        catalogue::OPERATIONS,
        chat::OPERATIONS,
        decide::OPERATIONS,
        file::OPERATIONS,
        fixture::OPERATIONS,
        inference::OPERATIONS,
        lora::OPERATIONS,
        model::OPERATIONS,
        onboarding::OPERATIONS,
        pairs::OPERATIONS,
        request::OPERATIONS,
        runtime::OPERATIONS,
        star::OPERATIONS,
        sources::OPERATIONS,
        steering::OPERATIONS,
        tensor::OPERATIONS,
        tokenizer::OPERATIONS,
        tune::OPERATIONS,
        workflow::OPERATIONS,
        workspace::OPERATIONS,
    ]
    .concat();
    all.extend(telemetry::operations());
    all
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

/// `error` with `context` (what the step had reached or left: `pair 3`,
/// `the output it created was removed`) added to its declared sentence, so
/// the refusal keeps its cause, its exit status and its fix and says this
/// too. An error no cause declares yet keeps `context` as its context line.
pub fn amend(error: anyhow::Error, context: impl fmt::Display) -> anyhow::Error {
    match error.downcast::<Declared>() {
        Ok(mut declared) => {
            declared.sentence = format!("{}; {context}", declared.sentence);
            anyhow::Error::new(declared)
        }
        Err(error) => error.context(context.to_string()),
    }
}

/// The refusal another Wisent library declared and handed over as its
/// envelope (`wisent_errors::Failure`, its `context["cause"]` naming one of
/// `operations`' causes): that cause, so the exit status, the id and the fix
/// stay the declaring library's, with the envelope's own sentence and fix
/// and `step` (what Ster was doing) after them. An envelope naming no cause
/// `operations` declare is `undeclared`'s refusal, quoting the envelope.
pub fn relay(
    failure: &wisent_errors::Failure,
    operations: &[&'static Operation],
    undeclared: &'static Cause,
    step: &str,
) -> anyhow::Error {
    let named = failure.context.get("cause");
    let cause = named.and_then(|id| {
        operations
            .iter()
            .flat_map(|operation| operation.causes.iter().copied())
            .find(|cause| cause.id == id)
    });
    // The envelope's own values, so a cause whose envelope carries no
    // sentence or fix still reads in its declaration's words with them.
    let values: Vec<(&str, &str)> = failure
        .context
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    let detail = failure.detail.as_deref().filter(|detail| !detail.is_empty());
    let remedy = failure.remedy.as_deref().filter(|remedy| !remedy.is_empty());
    match cause {
        Some(cause) => anyhow::Error::new(Declared {
            cause,
            sentence: match detail {
                Some(detail) => format!("{detail}; it was {step}"),
                None => format!(
                    "{}; {} handed over no sentence for it; it was {step}",
                    cause.sentence(&values),
                    failure.failure_point
                ),
            },
            fix: match remedy {
                Some(remedy) => remedy.to_owned(),
                None => cause.fix(&values),
            },
            underneath: None,
        }),
        None => refuse(
            undeclared,
            &[
                ("step", step),
                ("point", &failure.failure_point),
                ("code", failure.code.as_str()),
                ("named", named.map_or("no cause", String::as_str)),
                ("detail", match detail { Some(detail) => detail, None => "no sentence" }),
                ("remedy", match remedy { Some(remedy) => remedy, None => "no fix" }),
            ],
        ),
    }
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
    let findings: Vec<&str> = answer["findings"]
        .as_array()
        .map_or_else(Vec::new, |all| all.iter().filter_map(Value::as_str).collect());
    (!findings.is_empty()).then(|| {
        refuse(
            &catalogue::FINDINGS,
            &[
                ("count", &findings.len().to_string()),
                ("findings", &findings.join("; ")),
            ],
        )
    })
}

/// The status a command that ends in `error` exits with: the declared
/// cause's, or [`FAILURE_EXIT`] for an error no cause declares yet.
pub fn exit_code(error: &anyhow::Error) -> i32 {
    declared(error).map_or(FAILURE_EXIT, Declared::exit_code)
}

