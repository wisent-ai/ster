# Unreleased: steering workflows, tokenizer declarations, the star request and telemetry

Part of the [changelog](../../../CHANGELOG.md).

- Every steering workflow declares how it fails
  (`docs/guide/errors/workflow.md`): choosing `--layers`
  (`ster.vector.layers`: a bound or layer that is not a number, a range of
  no layer, a selection of no layer and every layer past the model, named
  together; now usage refusals, exit 2, which used to exit 1), reading a
  prompt set (`ster.prompts`), fitting and scoring a direction
  (`ster.vector.direction`: an unknown `--method`, reads that are not finite,
  naming the side, pair and value, and two sides that read alike), the
  holdout of `optimize` and `curve` (`ster.vector.holdout`, `ster.vector.curve`),
  `vector compare`, `vector project`, `vector score`, `extract` and `parity`
  (one cause per record field, tensor and shape it refuses).
- The pair set, the prompt set, the holdout and `--sizes` are read and
  checked before any weight is mapped, so a wrong one costs no checkpoint
  load; `vector compare` refuses one artifact or more groups than artifacts
  before reading any; `vector score` refuses an empty or missing text before
  the artifact. `vector evaluate` checks the artifact against the model the
  way every steered command does.
- `ster extract` refuses a state that is not finite instead of writing it as
  `null`; `vector compare` with `"layers": []` compares every shared layer
  instead of none.
- `ster model tokenizer create` and the tokenizers a fixture composes declare
  how they fail (`docs/guide/errors/tokenizer.md`): each refusal names the
  path, the token and its position; a repeated token names both positions; a
  post-processor emitting an id the vocabulary lacks names the token; a run
  that fails removes its output and says so, or says what is left and the
  `rm` that removes it.
- `ster converse` (`ster.converse`), `ster star` and `ster request star/*`
  (`ster.star`, one cause per kind the onboarding runtime answers, naming the
  state file, XDG_STATE_HOME, HOME and USER) are catalogued;
  `tests/errors/uncatalogued.json` is empty.
- `ster telemetry` and `ster request telemetry/*` keep the cause
  wisent-errors' telemetry declared (its id, exit status and fix) and say
  which command it was; a detached flush that cannot start is said on
  standard error.
- `ster telemetry show` and `ster request telemetry/show` answer
  `{"events": [...]}` where they answered a bare list, which the request
  protocol refuses as a result (Ster Desktop's **Show what waits** was
  refused `event-shapeless`); `tests/telemetry/telemetry.sh` reads both,
  and it and `tests/support/support.sh` now run the binary under the run's
  own home.
- A refusal raised under another step keeps its declared cause and gains
  that step (`it was pair 3`, `the output this run created was removed`)
  instead of a free-text context line.
- Many refusals whose observed state was fixed text now carry the values of
  their site: the renderings a chat template disagreed on, the CSV line a
  quote opened on, the trait a synthesis writes, the scorer a bearer is for,
  whether HOME is unset or empty, and the like.
- Real tests: `tests/steering/{layers,direction,holdout,workflow,parity}.sh`,
  `tests/models/declaration.sh`, `tests/onboarding/star.sh`, the named
  refusals of `tests/steering/score.sh`, and `tests/tuning/objectives.sh`'s
  one-turn conversation; `tests/tuning/provenance.mjs` asserts the declared
  sidecar refusals.
