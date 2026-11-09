//! How `ster errors` itself fails.

use wisent_errors::{Cause, Code, Operation};

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

pub(super) static BASELINE_UNREADABLE: Cause = Cause {
    id: "baseline-unreadable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the {what} baseline {path}",
    cause: "the file could not be read: {error}",
    observed: "baseline: {path}",
    remedy: "name the baseline file the repository holds: tests/errors/uncatalogued.json for --uncatalogued, tests/errors/untested.json for --untested",
    test: "tests/errors/catalogue.sh::check_refuses_an_unreadable_baseline",
};

pub(super) static BASELINE_MALFORMED: Cause = Cause {
    id: "baseline-malformed",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the {what} baseline {path}",
    cause: "it is not a JSON list of strings: {error}",
    observed: "baseline: {path}",
    remedy: "write it as a JSON list, e.g. [\"generate\"] of command names for --uncatalogued or of declared test paths for --untested; `ster errors check` prints both lists",
    test: "tests/errors/catalogue.sh::check_refuses_a_baseline_that_is_not_a_list",
};

pub(super) static FINDINGS: Cause = Cause {
    id: "findings",
    code: Code::Refused,
    usage: false,
    exit: None,
    failed: "checking the failure catalogue",
    cause: "{count} finding(s): a malformed declaration, a declared test the repository does not hold, a command no functionality names, or a baseline entry no longer needed",
    observed: "the findings, each listed in the answer printed above",
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
    uses: &[],
    summary: "list, show and check every declared way a Ster functionality fails",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &FUNCTIONALITY_UNKNOWN,
        &PAGE_UNKNOWN,
        &BASELINE_UNREADABLE,
        &BASELINE_MALFORMED,
        &FINDINGS,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&ERRORS];
