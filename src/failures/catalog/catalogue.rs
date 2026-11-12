//! How `ster errors` itself fails.

use serde_json::{Value, json};
use wisent_errors::{Cause, Code, Operation};

use super::{DOCS as ERRORS_DOCS, FAILURE_EXIT, operations, refuse};

/// docs/guide/errors/catalogue.md, where this functionality is listed.
const DOCS: &str = page!("catalogue");

pub(super) static PAGE_UNKNOWN: Cause = Cause {
    id: "page-unknown",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: "printing the catalogue sections of {page}",
    cause: "no functionality is listed on {page}",
    observed: "pages: {pages}",
    remedy: "name one of those pages, or leave --page out for every section",
    test: "tests/errors/catalogue.sh::markdown_refuses_a_page_no_functionality_is_listed_on",
};

pub(super) static FUNCTIONALITY_UNKNOWN: Cause = Cause {
    id: "functionality-unknown",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: "showing the failure catalogue of {id}",
    cause: "no functionality of this ster declares a catalogue under {id}",
    observed: "declared functionalities: {declared}",
    remedy: "name one of the declared functionalities; `ster errors list` prints them",
    test: "tests/errors/catalogue.sh::show_refuses_an_undeclared_functionality",
};

/// What a baseline holds, for the `ster.file` refusals that name it.
pub(super) const BASELINE_FORM: &str = "a JSON list of strings, as tests/errors/uncatalogued.json and tests/errors/untested.json are";

pub(super) static BASELINE_MISSING: Cause = Cause {
    id: "baseline-missing",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: "reading {role} {path}",
    cause: "no file is there: nothing at {path} is the baseline named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name the baseline file the repository holds: tests/errors/uncatalogued.json for --uncatalogued, tests/errors/untested.json for --untested",
    test: "tests/errors/catalogue.sh::check_refuses_an_unreadable_baseline",
};

pub(super) static FINDINGS: Cause = Cause {
    id: "findings",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "checking the failure catalogue",
    cause: "{count} finding(s): a malformed declaration, a declared test the repository does not hold, a command no functionality names, or a baseline entry no longer needed",
    observed: "findings: {findings}",
    remedy: "fix each finding as it says: correct the declaration in src/failures, write the declared test, declare the command's failures, or remove the entry from its baseline",
    test: "tests/errors/catalogue.sh::check_finds_a_baseline_entry_no_longer_needed",
};

/// Listing, showing and checking the failure catalogue.
pub(super) static ERRORS: Operation = Operation {
    id: "ster.errors",
    entries: &[
        "ster errors list",
        "ster errors show",
        "ster errors check",
        "ster request errors/list",
        "ster request errors/show",
    ],
    uses: &[&super::file::FILE],
    summary: "list, show and check every declared way a Ster functionality fails",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &FUNCTIONALITY_UNKNOWN,
        &PAGE_UNKNOWN,
        &BASELINE_MISSING,
        &FINDINGS,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&ERRORS];

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
                &FUNCTIONALITY_UNKNOWN,
                &[("id", id), ("declared", &declared)],
            ))
        }
    }
}

fn document(operation: &Operation) -> anyhow::Result<Value> {
    // wisent_errors' `Operation::to_json` is `format!("{{\"operation\":{},
    // ...}}", escape(self.id), ...)` over the catalogue's static strings,
    // every one passed through `escape`, so its text is always a document.
    Ok(serde_json::from_str(&operation.to_json(FAILURE_EXIT))
        .expect("Operation::to_json writes a JSON document"))
}

/// `ster errors list`: every functionality with its entries, what it uses
/// and the causes it declares.
pub fn list() -> anyhow::Result<Value> {
    let operations = operations()
        .into_iter()
        .map(document)
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(json!({"docs": ERRORS_DOCS, "functionalities": operations}))
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
                &PAGE_UNKNOWN,
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
