# How a desktop request fails before its operation runs

Every declared way `ster request <operation>` refuses before the operation
it names runs: naming an operation it serves, reading the JSON body on
standard input into that operation's fields, the fields each operation checks
before a model loads, and the standard output its NDJSON events go to. What
fails once the operation runs is that operation's own page; the operations,
their bodies and the statuses a request exits with are on
[desktop requests](../desktop-requests.md). How a refusal reads and which
status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/request.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.request`

name an operation `ster request` serves, read its JSON body into the operation's fields, and check the fields it needs before a model loads

Started through: `ster request`. Uses: none. Failure catalogue `ster.request`: 11 declared cause(s) of its own.

<a id="ster-request-operation-unknown"></a>

### `ster.request.operation-unknown`

- **What failed:** starting request {operation}
- **Cause:** unknown operation: {operation} is not an operation `ster request` serves
- **Observed state shown:** operation: {operation}; served: {served}
- **Fix:** name one of the served operations, spelled as docs/guide/desktop-requests.md lists them
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::an_unknown_operation_is_refused`

<a id="ster-request-body-not-json"></a>

### `ster.request.body-not-json`

- **What failed:** reading the body of request {operation}
- **Cause:** request body does not match the operation schema: it is not JSON
- **Observed state shown:** {size} bytes; the JSON reader stopped at line {line}, column {column}: {error}
- **Fix:** send the body docs/guide/desktop-requests.md gives for {operation}: one JSON object with the fields it names, in camelCase
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_body_that_is_not_json_is_refused`

<a id="ster-request-body-truncated"></a>

### `ster.request.body-truncated`

- **What failed:** reading the body of request {operation}
- **Cause:** request body does not match the operation schema: it ends before its JSON value closes, so the sender stopped writing it midway
- **Observed state shown:** {size} bytes; the JSON reader ran out at line {line}, column {column}: {error}
- **Fix:** send the body docs/guide/desktop-requests.md gives for {operation}: one JSON object with the fields it names, in camelCase
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_body_cut_short_is_refused`

<a id="ster-request-body-shape"></a>

### `ster.request.body-shape`

- **What failed:** reading the body of request {operation}
- **Cause:** request body does not match the operation schema: a field {operation} requires is missing, a field it does not take is present, or a field holds another type or value than it takes
- **Observed state shown:** {size} bytes; at line {line}, column {column}: {error}
- **Fix:** send the body docs/guide/desktop-requests.md gives for {operation}: one JSON object with the fields it names, in camelCase
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_body_of_another_shape_is_refused`

<a id="ster-request-field-blank"></a>

### `ster.request.field-blank`

- **What failed:** checking the {operation} request
- **Cause:** {operation} requires {what}: the request's {field} field is absent or blank
- **Observed state shown:** {field}: {value}
- **Fix:** give the request's {field} field {what}, as the CLI's {flag} takes it
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_request_without_a_required_field_is_refused`

<a id="ster-request-list-short"></a>

### `ster.request.list-short`

- **What failed:** checking the {operation} request
- **Cause:** {operation} requires at least {least} {what}: the request's {field} list holds {count}
- **Observed state shown:** {field}: {count} entries
- **Fix:** give the request's {field} list at least {least} {what}, as the CLI's {flag} takes them
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_list_shorter_than_its_operation_needs_is_refused`

<a id="ster-request-field-unpaired"></a>

### `ster.request.field-unpaired`

- **What failed:** checking the {operation} request
- **Cause:** {field} requires {other}: the request gives {field} and leaves {other} out, and one means nothing without the other
- **Observed state shown:** {field}: {value}; {other}: absent
- **Fix:** give the request's {other} field as well, or leave {field} out
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_field_without_its_partner_is_refused`

<a id="ster-request-stdout-discarded"></a>

### `ster.request.stdout-discarded`

- **What failed:** starting request {operation}
- **Cause:** standard output reaches no reader: the process that started `ster request` closed it or sent it to the null device, so the request's events and its result would go nowhere and nothing is run
- **Observed state shown:** standard output: {stdout}; standard input: {stdin}
- **Fix:** start the request with its standard output piped to the reader of its NDJSON events (`ster request {operation} < body.json | cat` prints them), or run the CLI command docs/guide/desktop-requests.md names for {operation} when no reader wants the events
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_standard_output_nobody_reads_is_refused`

<a id="ster-request-body-directory"></a>

### `ster.request.body-directory`

- **What failed:** reading the body of request {operation} from standard input
- **Cause:** standard input is a directory, not a stream of the body: the process that started `ster request` redirected a directory into it
- **Observed state shown:** standard input: {state}; error kind {kind}; the system answered: {error}
- **Fix:** redirect the body's JSON file into standard input (`ster request {operation} < body.json`) or pipe the body in
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/request/refusals.sh::a_body_from_a_directory_is_refused`

<a id="ster-request-body-unread"></a>

### `ster.request.body-unread`

- **What failed:** reading the body of request {operation} from standard input
- **Cause:** standard input broke while the body was read: the system refused a read of the stream the body comes on (a device that refuses reads, a stream torn down while it was written)
- **Observed state shown:** standard input: {state}; {size} bytes read before the read failed; error kind {kind}; the system answered: {error}
- **Fix:** start the request again with its body piped or redirected from a file into standard input, written whole before it is closed
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/request/refusals.sh::a_body_whose_stream_breaks_is_refused`

<a id="ster-request-event-unwritten"></a>

### `ster.request.event-unwritten`

- **What failed:** writing a {event} event to standard output
- **Cause:** nobody reads the request's standard output any more: the process that started `ster request` closed it or went away, so the request stops rather than run for no reader
- **Observed state shown:** event: {event}; error kind {kind}; the system answered: {error}
- **Fix:** keep reading the request's standard output until its result event, then start the request again
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/request/refusals.sh::an_event_nobody_reads_is_refused`
