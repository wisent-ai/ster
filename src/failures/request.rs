//! How a desktop request fails before its operation runs (src/request):
//! naming an operation `ster request` serves, reading its JSON body into the
//! operation's fields, and the fields each operation checks before it loads a
//! model. Each refusal is result status 2; what fails once the operation runs
//! is the operation's own cause.

use wisent_errors::{Cause, Code, Operation};

use super::refuse;

/// docs/guide/errors/request.md, where this functionality is listed.
const DOCS: &str = page!("request");

const BODY_REMEDY: &str = "send the body docs/guide/desktop-requests.md gives for {operation}: one JSON object with the fields it names, in camelCase";

/// The refusal of `operation`'s body, which serde_json stopped reading
/// with `error` after `size` bytes: not JSON, cut short, or of another
/// shape than the operation's fields.
pub fn body_refusal(operation: &str, size: usize, error: &serde_json::Error) -> anyhow::Error {
    use serde_json::error::Category;
    let cause = match error.classify() {
        Category::Syntax => &BODY_NOT_JSON,
        Category::Eof => &BODY_TRUNCATED,
        Category::Data | Category::Io => &BODY_SHAPE,
    };
    refuse(
        cause,
        &[
            ("operation", operation),
            ("size", &size.to_string()),
            ("line", &error.line().to_string()),
            ("column", &error.column().to_string()),
            ("error", &error.to_string()),
        ],
    )
}

/// Refuses `operation`'s request when its `field` (the CLI's `flag`), which
/// should hold `what`, is blank.
pub fn require(
    value: &str,
    operation: &str,
    field: &str,
    what: &str,
    flag: &str,
) -> anyhow::Result<()> {
    if !value.trim().is_empty() {
        return Ok(());
    }
    Err(refuse(
        &FIELD_BLANK,
        &[
            ("operation", operation),
            ("field", field),
            ("what", what),
            ("flag", flag),
            ("value", &format!("{value:?}")),
        ],
    ))
}

/// Refuses `operation`'s request that gives `field` (`value`) without the
/// `other` field it only works with.
pub fn unpaired(operation: &str, field: &str, value: &str, other: &str) -> anyhow::Error {
    refuse(
        &FIELD_UNPAIRED,
        &[
            ("operation", operation),
            ("field", field),
            ("value", value),
            ("other", other),
        ],
    )
}

/// Refuses `operation`'s request whose `field` list (the CLI's `flag`)
/// holds `count` of the `least` `what` it needs.
pub fn too_few(
    operation: &str,
    field: &str,
    what: &str,
    least: usize,
    count: usize,
    flag: &str,
) -> anyhow::Error {
    refuse(
        &LIST_SHORT,
        &[
            ("operation", operation),
            ("field", field),
            ("what", what),
            ("least", &least.to_string()),
            ("count", &count.to_string()),
            ("flag", flag),
        ],
    )
}

pub static OPERATION_UNKNOWN: Cause = Cause {
    id: "operation-unknown",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: "starting request {operation}",
    cause: "unknown operation: {operation} is not an operation `ster request` serves",
    observed: "operation: {operation}; served: {served}",
    remedy: "name one of the served operations, spelled as docs/guide/desktop-requests.md lists them",
    test: "tests/request/refusals.sh::an_unknown_operation_is_refused",
};

pub static BODY_NOT_JSON: Cause = Cause {
    id: "body-not-json",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the body of request {operation}",
    cause: "request body does not match the operation schema: it is not JSON",
    observed: "{size} bytes; the JSON reader stopped at line {line}, column {column}: {error}",
    remedy: BODY_REMEDY,
    test: "tests/request/refusals.sh::a_body_that_is_not_json_is_refused",
};

pub static BODY_TRUNCATED: Cause = Cause {
    id: "body-truncated",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the body of request {operation}",
    cause: "request body does not match the operation schema: it ends before its JSON value closes, so the sender stopped writing it midway",
    observed: "{size} bytes; the JSON reader ran out at line {line}, column {column}: {error}",
    remedy: BODY_REMEDY,
    test: "tests/request/refusals.sh::a_body_cut_short_is_refused",
};

pub static BODY_SHAPE: Cause = Cause {
    id: "body-shape",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the body of request {operation}",
    cause: "request body does not match the operation schema: a field {operation} requires is missing, a field it does not take is present, or a field holds another type or value than it takes",
    observed: "{size} bytes; at line {line}, column {column}: {error}",
    remedy: BODY_REMEDY,
    test: "tests/request/refusals.sh::a_body_of_another_shape_is_refused",
};

pub static FIELD_BLANK: Cause = Cause {
    id: "field-blank",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the {operation} request",
    cause: "{operation} requires {what}: the request's {field} field is absent or blank",
    observed: "{field}: {value}",
    remedy: "give the request's {field} field {what}, as the CLI's {flag} takes it",
    test: "tests/request/refusals.sh::a_request_without_a_required_field_is_refused",
};

pub static LIST_SHORT: Cause = Cause {
    id: "list-short",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the {operation} request",
    cause: "{operation} requires at least {least} {what}: the request's {field} list holds {count}",
    observed: "{field}: {count} entries",
    remedy: "give the request's {field} list at least {least} {what}, as the CLI's {flag} takes them",
    test: "tests/request/refusals.sh::a_list_shorter_than_its_operation_needs_is_refused",
};

pub static FIELD_UNPAIRED: Cause = Cause {
    id: "field-unpaired",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the {operation} request",
    cause: "{field} requires {other}: the request gives {field} and leaves {other} out, and one means nothing without the other",
    observed: "{field}: {value}; {other}: absent",
    remedy: "give the request's {other} field as well, or leave {field} out",
    test: "tests/request/refusals.sh::a_field_without_its_partner_is_refused",
};

pub static STDOUT_DISCARDED: Cause = Cause {
    id: "stdout-discarded",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "starting request {operation}",
    cause: "standard output reaches no reader: the process that started `ster request` closed it or sent it to the null device, so the request's events and its result would go nowhere and nothing is run",
    observed: "standard output: {stdout}; standard input: {stdin}",
    remedy: "start the request with its standard output piped to the reader of its NDJSON events (`ster request {operation} < body.json | cat` prints them), or run the CLI command docs/guide/desktop-requests.md names for {operation} when no reader wants the events",
    test: "tests/request/refusals.sh::a_standard_output_nobody_reads_is_refused",
};

pub static BODY_DIRECTORY: Cause = Cause {
    id: "body-directory",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the body of request {operation} from standard input",
    cause: "standard input is a directory, not a stream of the body: the process that started `ster request` redirected a directory into it",
    observed: "standard input: {state}; error kind {kind}; the system answered: {error}",
    remedy: "redirect the body's JSON file into standard input (`ster request {operation} < body.json`) or pipe the body in",
    test: "tests/request/refusals.sh::a_body_from_a_directory_is_refused",
};

pub static BODY_UNREAD: Cause = Cause {
    id: "body-unread",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "reading the body of request {operation} from standard input",
    cause: "standard input broke while the body was read: the system refused a read of the stream the body comes on (a device that refuses reads, a stream torn down while it was written)",
    observed: "standard input: {state}; {size} bytes read before the read failed; error kind {kind}; the system answered: {error}",
    remedy: "start the request again with its body piped or redirected from a file into standard input, written whole before it is closed",
    test: "tests/request/refusals.sh::a_body_whose_stream_breaks_is_refused",
};

pub static EVENT_UNWRITTEN: Cause = Cause {
    id: "event-unwritten",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "writing a {event} event to standard output",
    cause: "nobody reads the request's standard output any more: the process that started `ster request` closed it or went away, so the request stops rather than run for no reader",
    observed: "event: {event}; error kind {kind}; the system answered: {error}",
    remedy: "keep reading the request's standard output until its result event, then start the request again",
    test: "tests/request/refusals.sh::an_event_nobody_reads_is_refused",
};

/// Reading a desktop request: its operation, its body and the fields the
/// operation checks before it runs.
pub static REQUEST: Operation = Operation {
    id: "ster.request",
    entries: &["ster request"],
    uses: &[],
    summary: "name an operation `ster request` serves, read its JSON body into the operation's fields, and check the fields it needs before a model loads",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &OPERATION_UNKNOWN,
        &BODY_NOT_JSON,
        &BODY_TRUNCATED,
        &BODY_SHAPE,
        &FIELD_BLANK,
        &LIST_SHORT,
        &FIELD_UNPAIRED,
        &STDOUT_DISCARDED,
        &BODY_DIRECTORY,
        &BODY_UNREAD,
        &EVENT_UNWRITTEN,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&REQUEST];
