# CLI

Every Ster command, what the Python `wisent` CLI called it, and the guide each
command family has. The [README](../../../README.md) links here.

```text
ster vector train     learn one vector per selected layer
ster vector optimize  select method and layer on a held-out fraction the caller states
ster vector evaluate  measure a vector on a contrastive pair set
ster vector inspect   summarize and validate a steering artifact
ster vector compare   compare vectors fitted for different traits on one model
ster vector curve     score one method on growing numbers of pairs
ster vector project   show where a vector moves the model's states at one layer
ster vector ablate    write a vector's direction out of a checkpoint's weights
ster generate         run normal or steered generation, with any number of vectors
ster extract          export hidden states for an arbitrary prompt set
ster onboarding import or replay first use
ster decide     answer typed questions about a state from one forward pass
ster calibrate  fit the temperature that makes decision probabilities honest
ster workspace  import, activate, and inspect persistent pair sets
ster errors     list, show and check every declared way a command fails
ster request    run one desktop request to completion: JSON body on stdin, NDJSON events on stdout
```

Run `ster <command> --help` for exact arguments. Every answer is one JSON
document on stdout for machines; the global `--text` flag prints the same
answer as `path: value` lines for a person. A refusal names what failed, why,
the state it observed and what fixes it, and exits `2` for a usage refusal,
`69` when retrying unchanged can help and `1` otherwise; [How Ster
fails](../errors.md) lists every declared cause.

Ster replaced the Python `wisent` CLI, whose twenty-nine hyphenated verbs
each spelled an object into a command name. Each is now a verb of the object
it acts on, or a composition of them:

| wisent | Ster |
| --- | --- |
| `generate-pairs`, `synthetic` | `pairs synthesize` |
| `generate-pairs-from-task` | `pairs import` |
| `diagnose-pairs` | `pairs inspect` |
| `generate-vector`, `create-steering-vector` | `vector train` |
| `generate-vector-from-task`, `generate-vector-from-synthetic` | `pairs import` or `pairs synthesize`, then `vector train` |
| `diagnose-vectors` | `vector inspect` |
| `verify-steering` | `vector evaluate` |
| `compare-steering` | `vector compare` |
| `steering-viz` | `vector project` |
| `multi-steer` | `generate` with several `--vector`/`--strength` pairs |
| `generate-responses` | `generate --prompts` |
| `optimize-steering`, `find-best-method`, `discover-steering`, `optimize-classification` | `vector optimize` |
| `optimize-sample-size` | `vector curve` |
| `optimize` | `vector optimize`, then `vector curve` |
| `optimization-cache` | the artifact or report each run writes |
| `check-linearity` | `vector optimize`'s `holdout_effect_size` |
| `cluster-benchmarks` | `vector train` per benchmark, then `vector compare --clusters` |
| `geometry-search`, `train-unified-goodness` | `pairs merge`, then `vector optimize` or `vector train` |
| `zwiad` | `vector optimize`, `vector compare` and `vector project` |
| `modify-weights` | `vector ablate` |
| `get-activations` | `extract` |
| `tasks` | `pairs import` for the pairs, `decisions benchmark` for the score |
| `inference-config` | none: every sampling setting is a flag of the run that uses it |
| `agent` | none: autonomous work is Singularity's |

`evaluate-refusal` and `evaluate-responses` read what a model answered and
have no Ster form: whether an answer refused is a model's judgement, which
`pairs synthesize` asks the writing model for on every answer it writes, and
Ster keeps no list of refusal phrasings to score a file of answers by.

Each command family has its own page in this repository:

- [Pair sets](../pair-sets.md) — the file every command reads, and
  `ster pairs`, which authors it.
- [Steering](../steering.md) — choosing a direction, reading one, and
  what fitting one out of format costs.
- [Fine-tuning](../tuning/fine-tuning.md) — what `ster tune` trains, what a
  run needs before it starts, and what it records, with
  [objectives](../tuning/objectives.md),
  [adapters](../tuning/adapters.md),
  [chat templates](../tuning/chat-templates.md) and
  [precision](../tuning/precision.md) beside it.
- [Decisions](../decisions.md) — `ster decide`: what a decision
  is, the request and response documents, and reading one with `--explain`;
  with [calibration](../decisions/calibration.md) beside it.
- [Desktop requests](../desktop-requests.md) — `ster request`: how
  Ster Desktop runs each workflow as one process per operation, the JSON body,
  the NDJSON events, and what each exit status means.
- [How Ster fails](../errors.md) — `ster errors`: every declared cause of
  every functionality, its exit status and its test, one page per area.

## Telemetry, support and doctor

Ster reports nothing until you answer. The first command you run on a terminal
asks once whether it may send its maintainers its failures (`errors`), its
failures and the commands that ran (`usage`), or nothing; a script or a coding
agent is never asked. No file, prompt, pair or command argument is sent, and
your home directory and host name are taken out first. `ster telemetry
status|on|off|show|export|flush` reads and changes it and shows exactly what
waits to be sent, and `ster telemetry events` lists every kind of event Ster
can send, the consent each needs and every property it carries
([reference](https://ster.wisent.com/docs/cli/telemetry)); `DO_NOT_TRACK=1`
sends nothing.

`ster support report --description "what you did and what happened"
[--subject …] [--email …] [--without-failure]` asks Ster's support for help,
with Ster's version, the system's, this installation's id and the last failure
telemetry holds; `ster support status` prints every report this installation
filed with its status and replies
([reference](https://ster.wisent.com/docs/cli/support)). Ster Desktop's Help →
Report a Problem files and reads the same reports.

`ster doctor [--json]` tries every functionality Ster's failure catalogue
declares with an entry point, now, on this machine: the telemetry directory
can be written and where events go, where support reports go and their
replies, the workspace and its active pair set. Each is said to work with what
was seen, to fail with what failed and how to fix it, or to be skipped because
it changes your data; it exits 1 when one failed
([reference](https://ster.wisent.com/docs/cli/doctor)).
