//! How `ster telemetry` and `ster support` fail: through the functionalities
//! of wisent-errors' telemetry and support they run, which declare and test
//! their own causes.

use wisent_errors::Operation;
use wisent_errors::telemetry::{CONSENT, DELIVER, RECORD, REPORT, STATUS};

/// docs/guide/errors/telemetry.md, where this functionality is listed.
const DOCS: &str = page!("telemetry");

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
    causes: &[],
};

/// support.md in the error guide, where `ster support` is listed.
const SUPPORT_DOCS: &str = page!("support");

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
    causes: &[],
};

/// Ster's own functionalities first, then the ones they use, which a
/// catalogue lists so `ster errors show ster.telemetry` and `ster errors show
/// ster.support` reach their causes.
pub(super) static OPERATIONS: &[&Operation] = &[
    &TELEMETRY, &SUPPORT, &RECORD, &CONSENT, &DELIVER, &REPORT, &STATUS,
];
