# Tuning consumer qualification

These journeys execute the actual Ster CLI and native request frontend against
an explicitly declared local checkpoint. Export qualification also executes
vLLM's public batch interface on the actual exported adapter. No simulated
provider or owned Python driver stands in for that consumer.

## Candidate and isolation

Put inputs in an ignored environment file, such as
`.build/qualification.env`. Candidate inputs apply to every journey; training
and generation declarations apply where listed:

| Input | Used by | Meaning |
|---|---|---|
| `STER` | All | Absolute candidate executable path |
| `STER_TEST_SOURCE_REVISION` | All | Exact committed revision used to build that candidate |
| `STER_TEST_BINARY_SHA256` | All | Digest retained by the candidate build |
| `STER_TEST_MODEL_SPEC` | All | Absolute native model-fixture declaration |
| `STER_TEST_TUNING_SPEC` | BiPO, export, checkpoints | Absolute JSON file containing explicit native training parameters, keyed by objective |
| `STER_TEST_MAX_NEW_TOKENS` | BiPO, export | Declared generation budget for the consumer check |
| `STER_TEST_SEED` | BiPO, export | Declared consumer generation seed |

The runner refuses a different executable digest, source revision or dirty
checkout. HOME, XDG data/state/cache, temporary files and inference caches are
isolated inside the checkout's ignored build directory. The runner retains
commands, stdin, complete stdout/stderr, process statuses, native result events,
observed state and the final verdict before removing only its owned fixture.

Use Node with environment-file loading, `JSON.rawJSON`, `JSON.isRawJSON` and
JSON parse source contexts. Native numeric lexemes, including full-width
integer seeds, are preserved when parameter declarations or negative cases
are serialized; they are not rounded through JavaScript floating-point values.

## Training declaration

`STER_TEST_TUNING_SPEC` maps an objective name to its native camelCase
parameter object. Numbers are JSON numbers, not quoted CLI strings. See the
[required input matrix](../../desktop-requests.md#tuning-bodies-and-results).
The declaration supplies the experiment; no learning rate, seed, epoch count,
rank or other number is calculated from an unrelated dataset length.

For BiPO, declare `layer`, `strength`, `beta`, `epochs`, `learningRate`,
`accumulation`, `warmupSteps`, `maxSequence`, `batchSize` and `seed` under
`bipo`. Optional categorical settings have the same meanings as the CLI.
For SFT export qualification, declare the corresponding `sft` parameters,
including `rank`, `alpha` and explicit warmup.

Do not place model, dataset or output bindings in these parameter objects:
the journey binds them to its isolated actual artifacts. Checkpoint and resume
paths also belong to owned journey bindings, not an operator-data profile.
A collision refuses by field name instead of silently replacing an input.

## BiPO direction and generation

Additionally declare `STER_TEST_PAIRS` as the absolute pair-set path.

```sh
node --env-file=.build/qualification.env tests/tuning/bipo.mjs
```

The journey creates the real checkpoint, trains the CLI direction, reads its
persisted layer and finite native width, and confirms that the declared profile
trained every supplied pair. A native `tune/bipo` run must produce the same
direction and final loss from identical inputs.

The saved direction then steers real CLI and native generation. Its answer must
actually differ from the unsteered answer on the fixture's declared verification
prompt, and both frontends must agree. A profile that produces no observable
intervention does not qualify merely because the command accepted the artifact.
The out-of-model layer boundary and omitted layer/warmup fields must refuse
without output or changes to completed artifacts and original inputs. Frozen
base files must remain unchanged. Reports are under `.build/tuning/bipo/`.

## Portable PEFT export

Additionally declare:

| Input | Meaning |
|---|---|
| `STER_TEST_EXAMPLES` | Absolute SFT example-set path |
| `STER_TEST_VLLM` | Absolute installed vLLM executable path |
| `STER_TEST_VLLM_SPEC` | JSON object of explicit string/number vLLM resource options, using camelCase field names |
| `STER_TEST_VLLM_TEMPLATE` | Existing chat-template file appropriate for the declared fixture and prompt |

The consumer resource declaration must state `maxLoraRank`, using a value the
installed consumer supports and that accommodates the trained adapter. Other
resource choices belong to that declaration. The journey binds model, tokenizer,
served identities, native fixture context, full precision, chat template and
batch input/output paths itself; conflicting settings are refused. Use an
actual compatible vLLM installation and machine, not a stand-in executable.

```sh
node --env-file=.build/qualification.env tests/tuning/export/creation.mjs
```

A real SFT run produces the source weights. CLI and native exports must write
portable artifacts with identical contents and accurate file receipts. The
actual vLLM [`run-batch`](https://docs.vllm.ai/en/latest/cli/run-batch/) interface
loads the exported PEFT adapter and answers paired base/adapted requests through
its documented chat-completion batch format. Both request identities must
complete successfully, name the requested model and produce an observably active
adapter effect. A successful process, existing files or an unchanged answer is
not enough.

The consumer executable digest, reported version, declared resource inputs,
complete streams and actual batch input/output are retained under
`.build/tuning/export/`. Missing dependencies, unsupported architecture/rank,
consumer errors or no observable intervention fail qualification with their
actual receipts; they are not replaced with successful metadata checks.
Wrong-model export and missing/unsupported formats must refuse without output,
and neither inference nor refusal may alter the completed adapter, export,
frozen base or operator inputs.

These are backend and external-consumer journeys, not Desktop recordings.
Desktop qualification must separately observe the Tune and Adapter menus,
operation-specific fields, actual execution, result metadata and complete
failure/log presentation on the native graphical surface.

## System-turn evaluation

Declare `STER_TEST_EXAMPLES` as an SFT example set with no existing system
turns, `STER_TEST_CHAT_CONFIG` as the actual tokenizer configuration publishing
the chat template, and `STER_TEST_SYSTEM_PROMPT` as the system instruction.
`STER_TEST_MAX_SEQUENCE` and `STER_TEST_BATCH_SIZE` are explicit scoring inputs,
not a context or batch size inferred from unrelated input counts.

```sh
node --env-file=.build/qualification.env tests/tuning/system.mjs
```

The journey creates actual bare and chat checkpoints, applies the declared
template only to its owned chat fixture, and evaluates the same completions
with and without the declared system turn through both frontends. The system
instruction must observably affect measured loss without increasing either
per-example or total completion-target token counts. Every input is scored,
original ordering is preserved, and an absent system field remains absent.

Blank system turns, a checkpoint with no template and explicit plain-text mode
must refuse. A missing scoring limit is a pre-execution schema refusal. The
original examples, template, generated dataset and frozen base files must remain
unchanged; actual reports and refusals are retained under `.build/tuning/system/`.

## Checkpoint lifecycles

The training declaration must contain actual `sft`, `dpo`, `reward`, `grpo`,
`bipo` and `decide` parameter objects, including each objective's required
warmup, turns or permutations fields. Declare `STER_TEST_EXAMPLES`,
`STER_TEST_PAIRS`, `STER_TEST_PROMPTS` and `STER_TEST_DECISIONS` as their real
dataset paths. `STER_TEST_CHECKPOINT_EVERY` is the explicit save interval;
`STER_TEST_DIFFERENT_SEED` is a valid seed different from SFT's declared seed,
used for the mismatched-run refusal.

```sh
node --env-file=.build/qualification.env tests/tuning/checkpoints.mjs
```

Every objective first completes actual uninterrupted training. Its checkpointed
run is then stopped only after a filesystem observer sees the published native
`latest` pointer; no progress sentence, polling interval, sleep or timeout is
used as publication evidence. The process must actually end from the requested
signal, leave no completed output, and retain a saved step before the observed
total. Profiles that finish before interruption or have no intermediate save
do not qualify.

The native request resumes that persisted run and must reproduce the final
trained tensors and final loss, with the observed saved step reported as its
resume point. A valid unreferenced legacy-named attempt must remain unchanged.
Both interfaces then exercise changed-run, incomplete-setting, missing-state
and escaping-pointer refusals. The traversal target is a real valid copied
checkpoint outside the checkpoint root, so an absent target cannot explain
the refusal. Neither refusal may change that outside state or replace the
invalid pointer.

Commands, complete streams, publication observation, signal/status, saved
optimizer state, source-input digests and final tensor comparisons are retained
under `.build/tuning/checkpoints/`. A source declaration, live child or checkpoint
filename alone is not proof of a resumable lifecycle.
