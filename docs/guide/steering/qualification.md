# Steering qualification

These permanent journeys run Ster's actual CLI and native request interface
with a real locally created checkpoint. Their shared runner binds the run to
the declared committed source revision and executable digest, isolates HOME
and XDG data/state directories, retains commands, stdin, stdout, stderr, exit
statuses and observed artifacts, and removes only its owned fixture state.

## Inputs

Provide the candidate and experiment in an ignored environment file:

- `STER`: absolute candidate executable path.
- `STER_TEST_SOURCE_REVISION`: full committed source revision of that artifact.
- `STER_TEST_BINARY_SHA256`: executable digest from the qualification build.
- `STER_TEST_MODEL_SPEC`: absolute native fixture declaration path; the
  checked-in `docs/examples/model/fixture.json` is an explicit reference input,
  not a preset inferred by a command.
- `STER_TEST_SEED`: the experiment's declared generation/import seed.

Prompt and mixture journeys additionally require
`STER_TEST_MAX_NEW_TOKENS`. Their zero temperature is the native argmax
boundary, allowing independently generated answers to be compared without
substituting a token budget or seed derived from unrelated input counts.
Use Node with environment-file and `JSON.rawJSON` support; native integer
request inputs must not be rounded through JavaScript floating-point numbers.

The mixture journey also requires `STER_TEST_STRENGTH`: finite, nonzero and
with a finite sum when duplicated. The declared fixture prompt and strength
must actually change the doubled-steering answer from the unsteered answer.
An experiment that observes no active steering does not qualify cancellation
or duplicate-vector composition merely because every answer is equal.

## Prompt batches

```sh
node --env-file=.build/qualification.env tests/generate/prompts.mjs
```

The journey reads the persisted batch and compares every ordered prompt and
answer with independent single-prompt CLI and native request executions. An
empty set must fail without writing output or changing the completed batch.
No refusal wording or chosen sampled sentence is pinned. Reports are under
`.build/generate/`.

## Mixtures

```sh
node --env-file=.build/qualification.env tests/steering/mixed.mjs
```

Actual CAA directions are fitted to the checked-in pairs and their reversed
sides. Equal-strength opposite directions must cancel; duplicate directions
must compose to the sum of their strengths. The native request must produce
the same mixture answer. Missing and all-zero strengths are exercised through
the corresponding CLI/runtime and request-schema refusals. Learned artifacts
and original inputs must remain unchanged. Reports are under
`.build/steering/mixed/`.

## Pair and activation counts

```sh
node --env-file=.build/qualification.env tests/pairs/counts.mjs
```

Real BiPO CSV imports with unanswered rows and insufficient yield must refuse
without output. Empty pair and prompt sets must also refuse. A successful
extraction must persist each prompt in order with exactly every requested
model layer, finite vectors and the declared native width. Actual direction
training and `vector inspect` must agree on every persisted layer and width.
Reports are under `.build/pairs/counts/`.

## Strength selection

```sh
node --env-file=.build/qualification.env tests/steering/strength.mjs
```

Declare `STER_TEST_STRENGTHS` as comma-separated finite alternatives,
`STER_TEST_BATCH_SIZE` and `STER_TEST_MAX_SEQUENCE` in the environment file.
The compatible reference fixture must encode the sample pair sides beyond
the shortest valid sequence, so the all-too-long refusal is actually exercised.
The journey checks actual candidate order, unique selection, ordered-share
then mean-shift ranking, scored/skipped accounting and retained representation
measurements through both interfaces. Missing dependencies, nonfinite strength,
zero batch and too-short sequence are refusals, not wording snapshots.
Reports are under `.build/steering/strength/`.

## Merged pair sets

```sh
node --env-file=.build/qualification.env tests/pairs/merge.mjs
```

Every persisted pair and source receipt must agree with both input sets in
order. The merged dataset then trains a real direction consumed by native
inspection. Too few sources, blank traits and invalid request output are
refused without creating artifacts; completed merges and learned inputs remain
unchanged. Reports are under `.build/pairs/merge/`.

BiPO optimization and portable-adapter consumer journeys are documented under
[tuning qualification](../tuning/qualification/consumers.md).

These journeys exercise the native request backend, not the Desktop surface.
Desktop qualification separately observes actual actions and the result/log
panels; a source declaration or a passing backend journey is not a GUI recording.

## Permanent ablation

Declare `STER_TEST_PAIRS`, `STER_TEST_ABLATION_STRENGTH`,
`STER_TEST_MAX_NEW_TOKENS` and the generation seed alongside the candidate and
fixture inputs.

```sh
node --env-file=.build/qualification.env tests/vectors/ablate/lifecycle.mjs
```

The journey fits real CAA directions and rewrites actual checkpoint tensors
through CLI and native ablation. Native F32 payload inspection must show that
exactly the residual-stream writes for the learned layers changed, every
unrelated tensor stayed identical, and the native and CLI outputs agree.
The rewritten checkpoint must produce an observable generation effect and be
consumed equivalently through the native interface. No-op strength,
incompatible model and omitted strength refuse without output or changes to
the source model, learned artifact or pair set. Reports are retained under
`.build/vectors/ablate/`.

## Comparison geometry

This journey needs the candidate, fixture declaration and `STER_TEST_PAIRS`,
not a sampled-generation seed or budget.

```sh
node --env-file=.build/qualification.env tests/vectors/compare/geometry.mjs
```

Actual CAA, reversed-CAA and PCA directions are trained independently.
Comparison must preserve their identity/order and shared layers, identify the
opposite signed directions, place them on the same sign-blind axis and produce
finite geometry matrices. At the maximum valid group count, measured from the
actual artifact set, every artifact is a singleton and silhouette is undefined.
The native interface must agree on the actual geometry. Too few artifacts,
the next count beyond the grouping maximum and an absent layer are meaningful
refusals; learned inputs remain unchanged. Reports are retained under
`.build/vectors/compare/`.

## Learning curves

Declare `STER_TEST_HOLDOUT` as a native JSON number,
`STER_TEST_CURVE_SIZES` as a JSON array of positive integral pair counts,
`STER_TEST_CURVE_METHOD` as the actual direction method and `STER_TEST_PAIRS`.
Sizes and the rounded holdout must leave valid fitting and held-out sets;
they are experiment declarations, never fractions or presets synthesized from
the data size.

```sh
node --env-file=.build/qualification.env tests/vectors/curve/sampling.mjs
```

Every model layer must retain the declared sample order and the actual split
counts. The reported sufficient size must be the smallest declared size that
reached that layer's best measured accuracy. Independently persisted prefix
fits are evaluated on the actual held-out suffix and must reproduce the curve
scores. Native and CLI measurements must agree. Oversized samples, excluded
fraction endpoints and an omitted sample list refuse. Reports are retained
under `.build/vectors/curve/`.

## Projection states

Declare `STER_TEST_PAIRS`, `STER_TEST_PROJECTION_LAYER`,
`STER_TEST_PROJECTION_STRENGTH` and `STER_TEST_PROJECTION_COMPONENTS`.
Each projection parameter is an explicit native JSON number; the declared
experiment must observably move the measured negative states.

```sh
node --env-file=.build/qualification.env tests/vectors/project/geometry.mjs
```

Actual CAA directions are fitted at every model layer. Projection must retain
every source pair field and steering-state combination exactly once, produce
finite coordinates on the fitted components within the declared budget, and
report actual direction shifts and changed negative-state distances. Native
and CLI geometry must agree. An absent layer, an empty component budget and
omission of each required projection parameter refuse; schema refusals must
not enter native execution. Reports are retained under
`.build/vectors/project/`.

## Held-out selection

Declare `STER_TEST_PAIRS` as the source pair file and `STER_TEST_HOLDOUT`
as the explicit native fraction to rank candidates on.

```sh
node --env-file=.build/qualification.env tests/vectors/optimize/selection.mjs
```

The journey independently trains and consumes actual CAA, PCA and logistic
artifacts. Their measured layer/method results on the held-out suffix must
match every published candidate; signed effect sizes must agree with signed
mean margins when defined. Exactly the highest-accuracy candidate, then the
highest-margin tie winner, must be selected. Its persisted direction must
equal an independent refit on all input pairs. Native selection must agree,
and the actual native evaluation consumer must read the selected artifact.
Excluded fraction endpoints and an omitted holdout refuse without publishing
an artifact. Source models and operator inputs remain unchanged. Reports are
retained under `.build/vectors/optimize/`.
