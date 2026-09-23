# Desktop requests

`ster request <operation>` runs one JSON request to completion and exits. It is
how Ster Desktop runs every workflow: one `ster request` process per operation,
started when the operator presses the button and gone when the result arrives.
Nothing of Ster stays resident between two operations, and no port is opened.

```text
ster request <operation>   < request body (one JSON document on stdin)
```

Every operation runs the same functions as its CLI command, so the document in
the result is the one the command prints and the defaults are the ones its
flags have. The body carries each flag as a camelCase field; an empty body is
the empty object, so every field takes its default.

## Operations

| Operation | CLI command it mirrors |
|---|---|
| `train`, `optimize`, `evaluate`, `generate`, `extract`, `inspect` | the [steering](steering.md) commands |
| `decide`, `calibrate` | [`ster decide`](decisions.md), [`ster calibrate`](decisions/calibration.md) |
| `workspace/import-pairs` | `ster workspace import-pairs` |
| `pairs/inspect`, `pairs/save`, `pairs/synthesize` | the [pair-set](pair-sets.md) commands |
| `tune/sft`, `tune/dpo`, `tune/reward`, `tune/grpo`, `tune/merge`, `tune/evaluate`, `tune/inspect` | the [adapter](tuning/adapters.md) commands |

`decide` takes its request document inline under `request`, and `pairs/save`
takes the pairs inline under `entries`, because the desktop composes both;
every other operation names its files by path.

## Events

stdout carries only NDJSON, one event per line, flushed as it is written:

```text
{"type":"log","stream":"stderr","chunk":"reading the state once (…)\n"}
{"type":"result","status":0,"json":{…}}
```

Zero or more `log` events carry the progress lines the command would have
written to stderr, in the order it wrote them. Exactly one `result` event comes
last. Its `status` is also the process's exit status:

| Status | Meaning | `json` |
|---|---|---|
| `0` | The operation completed. | The document the CLI command prints. |
| `1` | The operation started and failed. The last `log` event is `error: <sentence>`. | `{"error": "<sentence>"}` |
| `2` | The request was refused before anything ran. No `log` event precedes it. | `{"error": "<sentence>"}` |

Status `2` covers an unknown operation (`unknown operation: <name>`), a body
that does not parse into the operation's fields
(`request body is not valid JSON: <parser's reason>`), and every field
refusal the operation checks before loading a model, such as
`decide requires a model` or `choice question '<id>' needs at least two options`.

A reader that closes stdout ends the request at its next event: nobody is left
to receive the result, so a long training run does not continue for a window
that has already closed.

## Example

```bash
echo '{"artifact": "vectors/honesty.json"}' | ster request inspect
```

prints one `result` event whose `json` is the summary `ster inspect` prints,
and exits `0`. The same body with the operation misspelled prints
`{"type":"result","status":2,"json":{"error":"unknown operation: inspct"}}`
and exits `2`.

## Tests

[`tests/decide/request.rs`](../../tests/decide/request.rs) runs
`ster request decide` on the built binary with the body Ster Desktop's Decide
screen composes, checks the answer, the progress log events and the exit
status, and checks that a malformed question is refused with its exact sentence
as a single status-`2` result before any model loads.
