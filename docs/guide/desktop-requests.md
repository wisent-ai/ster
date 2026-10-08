# Desktop requests

`ster request <operation>` runs one JSON request to completion and exits. It is
how Ster Desktop runs every workflow: one `ster request` process per operation,
started when the operator presses the button and gone when the result arrives.
Nothing of Ster stays resident between two operations, and no port is opened.

```text
ster request <operation>   < request body (one JSON document on stdin)
```

Every operation runs the same functions as its CLI command, so the document in
the result is the one the command prints, and a field is required exactly when
its flag is. The body carries each flag as a camelCase field. Every training,
sampling and judging number is required on both sides — Ster assumes none — so
a body that leaves one out is refused before anything runs, naming it:
`request body is not valid JSON: missing field \`learningRate\` at line <L> column <C>`.

## Operations

| Operation | CLI command it mirrors |
|---|---|
| `vector/train`, `vector/optimize`, `vector/curve`, `vector/evaluate`, `vector/inspect`, `vector/compare`, `vector/project`, `generate`, `extract` | the [steering](steering.md) commands (`vector/curve` takes `pairs`, `method`, `holdout`, `sizes` and optional `layers`; `vector/compare` takes `artifacts`, two or more paths, `clusters` and optional `layers`; `vector/project` takes `pairs`, `vector`, `layer`, `strength` and `components`; `generate` takes its steering as `steering: [{"vector", "strength"}]`) |
| `decide`, `calibrate` | [`ster decide`](decisions.md), [`ster calibrate`](decisions/calibration.md) |
| `workspace/import-pairs`, `workspace/show`, `workspace/select`, `workspace/remove` | `ster workspace import-pairs`, `show`, `select`, `remove` (`select` and `remove` take the set's `id`; all three answer `ster workspace show`'s document) |
| `pairs/import`, `pairs/inspect`, `pairs/save`, `pairs/synthesize` | the [pair-set](pair-sets.md) commands (`pairs/import` takes `benchmark`, `source`, `output`, `seed` and optional `examples`, `count`, `traitName`, and for `benchmark: "choices"` the row fields `question`, `choices`, `answer`, `answerForm` and `labels`; it answers `ster pairs import`'s document with the skipped-row report) |
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
echo '{"artifact": "vectors/honesty.json"}' | ster request vector/inspect
```

prints one `result` event whose `json` is the summary `ster vector inspect` prints,
and exits `0`. The same body with the operation misspelled prints
`{"type":"result","status":2,"json":{"error":"unknown operation: vector/inspct"}}`
and exits `2`.
