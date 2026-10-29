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
`request body does not match the operation schema: missing field \`learningRate\` at line <L> column <C>`.

## Operations

| Operation | CLI command it mirrors |
|---|---|
| `vector/train`, `vector/optimize`, `vector/curve`, `vector/evaluate`, `vector/inspect`, `vector/compare`, `vector/project`, `vector/ablate`, `generate`, `extract` | the [steering](steering.md) commands (`vector/curve` takes `pairs`, `method`, `holdout`, `sizes` and optional `layers`; `vector/compare` takes `artifacts`, two or more paths, `clusters` and optional `layers`; `vector/project` takes `pairs`, `vector`, `layer`, `strength` and `components`; `vector/ablate` takes `model`, `vector`, `strength` and `output`; `generate` takes its steering as `steering: [{"vector", "strength"}]`) |
| `decide`, `calibrate` | [`ster decide`](decisions.md), [`ster calibrate`](decisions/calibration.md) |
| `workspace/import`, `workspace/show`, `workspace/select`, `workspace/remove` | `ster workspace import`, `show`, `select`, `remove` (`select` and `remove` take the set's `id`; all three answer `ster workspace show`'s document) |
| `model/fixture/create`, `model/tokenizer/create` | `ster model fixture create`, `ster model tokenizer create`; each takes `from` and `output`, refuses unknown request fields, and returns its actual native artifact and consumer-verification report. See [fixtures](models/fixtures.md) and [tokenizers](models/tokenizers.md). |
| `pairs/import`, `pairs/inspect`, `pairs/merge`, `pairs/save`, `pairs/synthesize` | the [pair-set](pair-sets.md) commands (`pairs/import` takes `benchmark`, `source`, `output`, `seed` and optional `examples`, `count`, `traitName`, and for `benchmark: "choices"` the row fields `question`, `choices`, `answer`, `answerForm` and `labels`; it answers `ster pairs import`'s document with the skipped-row report; `pairs/merge` takes `sources`, two or more paths, `traitName` and `output`) |
| `tune/sft`, `tune/dpo`, `tune/bipo`, `tune/reward`, `tune/grpo`, `tune/decide`, `tune/merge`, `tune/export`, `tune/evaluate`, `tune/inspect` | the [training](tuning/objectives.md) and [adapter](tuning/adapters.md) commands; tuning bodies deserialize into the actual CLI argument types, not a second request schema |

`decide` takes its request document inline under `request`, and `pairs/save`
takes the pairs inline under `entries`, because the desktop composes both;
every other operation names its files by path.

### Tuning bodies and results

Each trainer requires `model`, its input path and `output`, plus
`learningRate`, `accumulation`, `warmupSteps`, `maxSequence` and `seed`.
Warmup is always explicit, including a run that wants no warmup.
The categorical defaults remain the declared CLI choices: CPU, full precision,
automatic chat formatting, all layers, query/value projections, DPO loss and
length reward where those fields apply.

| Operation | Additional required fields | Result |
|---|---|---|
| `tune/sft` | `examples`, `rank`, `alpha`, `epochs`, `batchSize` | `{"path": "...", "report": {...}}` |
| `tune/dpo` | `pairs`, `rank`, `alpha`, `beta`, `epochs`, `batchSize` | adapter path and preference report |
| `tune/bipo` | `pairs`, `layer`, `strength`, `beta`, `epochs`, `batchSize` | steering JSON path and BiPO report; no LoRA rank, alpha or projection fields |
| `tune/reward` | `pairs`, `rank`, `alpha`, `epochs`, `batchSize` | reward artifact path and scoring-head report |
| `tune/grpo` | `prompts`, `rank`, `alpha`, `group`, `iterations`, `beta`, `maxNewTokens`, `temperature`, `topP`, `turns` | policy adapter path and report; no epochs or batch size |
| `tune/decide` | `examples`, `rank`, `alpha`, `epochs`, `batchSize`, `permutations` | adapter path and decision report with labelled-question and expanded-row counts |
| `tune/merge` | `model`, `adapter`, `output` | `{"report": {...}}` |
| `tune/export` | `model`, `adapter`, `output`, `format: "peft"` | `{"report": {...}}`, including the written files and native PEFT metadata |
| `tune/evaluate` | `model`, `examples`, `maxSequence`, `batchSize` | the evaluation report directly; omit `adapter` for the bare checkpoint |
| `tune/inspect` | `artifact` | `{"adapter": {...}}`; no model is loaded |

Only the trainers take the common training numbers above. Every trainer also
accepts `checkpoint` together with `checkpointEvery`, and optional `resume`.
GRPO requires `turns`; decision training requires `permutations`. Neither
inherits an implicit numeric choice from a separate native frontend.
An unsupported export format, invalid typed precision/chat/device choice,
missing required field or trailing JSON is refused before execution.
A supplied blank path or a checkpoint setting without its counterpart is
also refused before execution. Model loading, training, artifact and filesystem
failures retain their actual chained cause once execution starts.

Ster Desktop's Tune menu exposes every trainer with its own input and output
kind. BiPO reports direction norm rather than invented LoRA parameters; decision
training reports questions and expanded rows. The Adapter menu includes PEFT
export and displays the files the backend reports writing. Completed results
retain the submitted objective even when the next form selection changes.


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
| `1` | The operation started and failed. The last `log` event is `error: <sentence>`. | `{"error": "<sentence>"}`, or the complete `ImportReport` for a rejected or conflicting workspace import. |
| `2` | The request was refused before anything ran. No `log` event precedes it. | `{"error": "<sentence>"}` |

Status `2` covers an unknown operation (`unknown operation: <name>`), a body
that does not parse into the operation's fields
(`request body does not match the operation schema: <parser's reason>`), and every field
refusal the operation checks before loading a model, such as
`decide requires a model` or `choice question '<id>' needs at least two options`.
An unknown `device`, `precision`, `chatTemplate` or `method` is a schema refusal
whose reason is the CLI's own sentence, for example
`request body does not match the operation schema: unknown device "tpu"; expected cpu, metal, or cuda at line <L> column <C>`.

`workspace/import` preserves the full domain report when it cannot accept the
set: `rejected` and `conflicting` are failed results, not successful imports.
The last log carries the report's reason. `imported` and `unchanged` complete
successfully and name the durable copy.

A reader that closes stdout ends the request at its next event: nobody is left
to receive the result, so a long training run does not continue for a window
that has already closed.

The result and process statuses retain the fixed identities declared by the
public [request protocol](https://ster.wisent.com/docs/request-api#events-and-exit-status).
They are not runtime overrides. If stdout cannot receive an event, stderr distinguishes
`Ster request cannot write its stdout event: <cause>` from
`Ster request cannot flush its stdout event: <cause>` and the process fails.
The cause is the actual I/O error, not a successful result.

## Example

```bash
echo '{"artifact": "vectors/honesty.json"}' | ster request vector/inspect
```

prints one `result` event whose `json` is the summary `ster vector inspect` prints,
and exits `0`. The same body with the operation misspelled prints
`{"type":"result","status":2,"json":{"error":"unknown operation: vector/inspct"}}`
and exits `2`.
