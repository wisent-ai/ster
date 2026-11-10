//! How `ster telemetry` fails: through the functionalities of wisent-errors'
//! telemetry it runs, which declare and test their own causes.

use wisent_errors::Operation;
use wisent_errors::telemetry::{CONSENT, DELIVER, RECORD};

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

/// Ster's own functionality first, then the ones it uses, which a catalogue
/// lists so `ster errors show ster.telemetry` reaches their causes.
pub(super) static OPERATIONS: &[&Operation] = &[&TELEMETRY, &RECORD, &CONSENT, &DELIVER];
