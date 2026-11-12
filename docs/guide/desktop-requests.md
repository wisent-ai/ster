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
`reading the body of request tune/sft: request body does not match the operation schema: a field tune/sft requires is missing, …; observed: <N> bytes; at line <L>, column <C>: missing field \`learningRate\` …`.

## Operations

| Operation | CLI command it mirrors |
|---|---|
| `vector/train`, `vector/optimize`, `vector/curve`, `vector/evaluate`, `vector/inspect`, `vector/compare`, `vector/project`, `vector/score`, `vector/ablate`, `generate`, `extract`, `parity` | the [steering](steering.md) commands (`vector/curve` takes `pairs`, `method`, `holdout`, `sizes` and optional `layers`; `vector/compare` takes `artifacts`, two or more paths, `clusters` and optional `layers`; `vector/project` takes `pairs`, `vector`, `layer`, `strength` and `components`; `vector/score` takes `vector`, `layer` and `text`; `vector/ablate` takes `model`, `vector`, `strength` and `output`; `generate` takes its steering as `steering: [{"vector", "strength"}]`; `parity` takes `model` and `input`) |
| `decide`, `calibrate` | [`ster decide`](decisions.md), [`ster calibrate`](decisions/calibration.md) |
| `workspace/import`, `workspace/show`, `workspace/select`, `workspace/remove` | `ster workspace import`, `show`, `select`, `remove` (`select` and `remove` take the set's `id`; all three answer `ster workspace show`'s document) |
| `model/fixture/create`, `model/tokenizer/create` | `ster model fixture create`, `ster model tokenizer create`; each takes `from` and `output`, refuses unknown request fields, and returns its actual native artifact and consumer-verification report. See [fixtures](models/fixtures.md) and [tokenizers](models/tokenizers.md). |
| `pairs/import`, `pairs/inspect`, `pairs/merge`, `pairs/save`, `pairs/synthesize` | the [pair-set](pair-sets.md) commands (`pairs/import` takes `benchmark`, `source`, `output`, `seed` and optional `examples`, `count`, `traitName`, and for `benchmark: "choices"` the row fields `question`, `choices`, `answer`, `answerForm` and `labels`; it answers `ster pairs import`'s document with the skipped-row report; `pairs/merge` takes `sources`, two or more paths, `traitName` and `output`) |
| `tune/sft`, `tune/dpo`, `tune/bipo`, `tune/reward`, `tune/grpo`, `tune/decide`, `tune/merge`, `tune/export`, `tune/evaluate`, `tune/inspect` | the [training](tuning/objectives.md) and [adapter](tuning/adapters.md) commands; tuning bodies deserialize into the actual CLI argument types, not a second request schema |
| `errors/list`, `errors/show` | `ster errors list`, `ster errors show` (`errors/show` takes the functionality's `id`); see [How Ster fails](errors.md) |
| `star/status`, `star/dismiss`, `star/reset` | `ster star status`, `dismiss`, `reset`; no fields; a refusal is `ster.star`'s cause, naming the state file |
| `doctor` | `ster doctor --json`: completes with `{"findings": [...], "status": 0 or 1}` even when a functionality fails, so the desktop's Diagnosis screen shows every finding |
| `telemetry/status`, `telemetry/set`, `telemetry/show`, `telemetry/flush`, `telemetry/events` | `ster telemetry status`, `on`/`off`, `show`, `flush`, `events`: what the Settings and What Ster Sends windows show and change; `telemetry/set` takes `level` (`off`, `errors` or `usage`); `telemetry/show` answers `{"events": [...]}`, every waiting event exactly as it will go, the same object `ster telemetry show` prints; `telemetry/events` answers `{"events": [...]}`, the list `ster telemetry events` prints; a refusal carries the cause wisent-errors' telemetry declared |
| `support/report`, `support/status` | `ster support report`, `status`; `support/report` takes `description` and optionally `subject`, `email` and `with_failure` (true unless given false, as without `--without-failure`); a refusal carries the cause wisent-errors' support client declared |

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
| `1` | The operation started and failed. The last `log` event is `error: <sentence>`. | `{"error": "<sentence>"}`, or the complete `ImportReport` for a rejected or conflicting workspace import; when a declared cause refused it, also `cause`. |
| `2` | The request was refused before anything ran. No `log` event precedes it. | `{"error": "<sentence>", "cause": {...}}` |

Status `2` is a cause of `ster.request` (the [request page](errors/request.md)
lists each with its sentence, fix and test) or a cause the operation declares
for its own settings:

- `operation-unknown`: an operation `ster request` does not serve; the
  sentence lists the ones it does.
- `body-directory`: a directory redirected into standard input
  (`ster request decide < some/dir`), observed with the directory's state and
  the system's answer.
- `body-unread`: standard input that breaks while the body is read; its
  `json.cause.exit` is `69`, since starting again with the body written whole
  can work.
- `body-not-json`, `body-truncated`, `body-shape`: a body that is not JSON,
  ends before its value closes, or does not fit the operation's fields (a
  missing or unknown field, another type); each says
  `request body does not match the operation schema` and observes the byte
  count, line, column and the parser's reason. An unknown `device`,
  `precision`, `chatTemplate` or `method` is `body-shape` whose reason is the
  CLI's own sentence, for example
  `unknown device "tpu"; expected cpu, metal, or cuda at line <L> column <C>`.
- `field-blank`: a field the operation needs, absent or blank, named with
  what it holds and the CLI flag that takes it:
  `checking the decide request: decide requires a model (a Hugging Face Hub id or a checkpoint directory): the request's model field is absent or blank`.
- `list-short`: a list with fewer entries than the operation needs
  (`vector/compare`'s `artifacts`, `vector/curve`'s `sizes`, `pairs/save`'s
  `entries`).
- `field-unpaired`: a field that means nothing without another
  (`vector/evaluate`'s `batchSize` and `maxSequence` without `strengths`,
  a tuning `checkpoint` without `checkpointEvery`).
- The operation's own settings, such as
  `choice question '<id>' needs at least two options` (`ster.decide.request`).

Two failures end the process without a result event, with the refusal on
stderr: `stdout-discarded` (status `2`), a standard output that was closed
(`>&-`, which Rust's runtime turns into the null device) or is the null device,
refused before standard input is read, so nothing runs for no reader; and
`event-unwritten` (the retryable status `69`), a standard output nobody reads
any more by the time an event is written.

`workspace/import` preserves the full domain report when it cannot accept the
set: `rejected` and `conflicting` are failed results, not successful imports.
The last log carries the report's reason. `imported` and `unchanged` complete
successfully and name the durable copy.

A failure a declared cause refused carries it in `json.cause`:
`{"functionality", "id", "docs", "code", "exit", "usage", "retryable", "remedy"}`,
where `functionality` is the id `ster errors show` (or `errors/show`) takes
for the cause's whole declaration (`ster.workspace.import`; `null` only for a
cause no listed functionality declares), `docs` the cause's line on that
functionality's area page (`…/docs/guide/errors/workspace.md#ster-workspace-import-name-conflict`),
`exit` the status the same CLI command exits with (`2` for a usage refusal,
`69` when retrying unchanged can help, `1` otherwise) and `remedy` the fix
with the actual values filled in. The request's own status stays `1`: the
operation had started. [How Ster fails](errors.md) lists every declared cause.

A reader that closes stdout ends the request at its next event: nobody is left
to receive the result, so a long training run does not continue for a window
that has already closed. The refusal names the event it could not write, the
system's error kind and its answer.

The result and process statuses retain the fixed identities declared by the
public [request protocol](https://ster.wisent.com/docs/request-api#events-and-exit-status).
They are not runtime overrides.

## Example

```bash
echo '{"artifact": "vectors/honesty.json"}' | ster request vector/inspect
```

prints one `result` event whose `json` is the summary `ster vector inspect` prints,
and exits `0`. The same body with the operation misspelled prints one result
event of status `2` whose `json.error` begins
`starting request vector/inspct: unknown operation: vector/inspct is not an operation \`ster request\` serves`
and whose `json.cause.id` is `operation-unknown`, and exits `2`.
