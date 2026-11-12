//! How `ster telemetry` and `ster support` fail: through the functionalities
//! of wisent-errors' telemetry and support they run, which declare and test
//! their own causes and hand each refusal back as an envelope naming its
//! cause ([`super::relay`]).

use wisent_errors::telemetry::{CONSENT, DELIVER, RECORD, REPORT, STATUS};
use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/telemetry.md, where this functionality is listed.
const DOCS: &str = page!("telemetry");

pub static UNDECLARED: Cause = Cause {
    id: "envelope-undeclared",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "wisent-errors' telemetry handed back a refusal naming no cause its functionalities declare, so its id, exit status and fix are unknown here",
    observed: "failure point {point}, code {code}, cause {named}: {detail}; its fix: {remedy}",
    remedy: "follow the fix it gives; build Ster against the wisent-errors revision whose telemetry declares that cause",
    test: "tests/telemetry/telemetry.sh::an_envelope_naming_no_declared_cause_is_refused",
};

pub static PROGRAM_UNNAMED: Cause = Cause {
    id: "program-unnamed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting the detached `ster telemetry flush` after `ster {command}`",
    cause: "the system does not name this program's executable, so no detached flush can be started from it: the binary was removed or replaced while it ran",
    observed: "error kind {kind}; the system answered: {error}; the events stay queued and the next run's flush sends them",
    remedy: "run `ster telemetry flush` from an installed ster; `ster telemetry show` lists what waits",
    test: "tests/telemetry/telemetry.sh::a_flush_without_its_program_is_said",
};

/// Reading and changing what this installation sends, and sending it.
pub(super) static TELEMETRY: Operation = Operation {
    id: "ster.telemetry",
    entries: &[
        "ster telemetry status",
        "ster telemetry on",
        "ster telemetry off",
        "ster telemetry show",
        "ster telemetry export",
        "ster telemetry flush",
        "ster request telemetry/status",
        "ster request telemetry/set",
        "ster request telemetry/show",
        "ster request telemetry/flush",
    ],
    uses: &[&RECORD, &CONSENT, &DELIVER],
    summary: "read and change what this installation of Ster reports to its maintainers, and send it",
    docs: DOCS,
    warnings: &[],
    causes: &[&UNDECLARED, &PROGRAM_UNNAMED],
};

/// support.md in the error guide, where `ster support` is listed.
const SUPPORT_DOCS: &str = page!("support");

pub static SUPPORT_UNDECLARED: Cause = Cause {
    id: "envelope-undeclared",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "wisent-errors' support client handed back a refusal naming no cause its functionalities declare, so its id, exit status and fix are unknown here",
    observed: "failure point {point}, code {code}, cause {named}: {detail}; its fix: {remedy}",
    remedy: "follow the fix it gives; build Ster against the wisent-errors revision whose support client declares that cause",
    test: "tests/support/support.sh::an_envelope_naming_no_declared_cause_is_refused",
};

/// Asking Ster's support for help and reading the replies.
pub(super) static SUPPORT: Operation = Operation {
    id: "ster.support",
    entries: &[
        "ster support report",
        "ster support status",
        "ster request support/report",
        "ster request support/status",
    ],
    uses: &[&REPORT, &STATUS],
    summary: "file a report with Ster's support from this installation and read the replies",
    docs: SUPPORT_DOCS,
    warnings: &[],
    causes: &[&SUPPORT_UNDECLARED],
};

/// Ster's own functionalities first, then every functionality of
/// wisent-errors' telemetry and support, which a catalogue lists so `ster
/// errors show ster.telemetry` and `ster errors show ster.support` reach their
/// causes and a relayed refusal finds its declaration.
pub fn operations() -> Vec<&'static Operation> {
    let mut listed = vec![&TELEMETRY, &SUPPORT];
    listed.extend(wisent_errors::telemetry::operations());
    listed
}
