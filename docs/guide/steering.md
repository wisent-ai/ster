# Steering

Choosing a direction, reading what it is, and what it costs to fit one out of
the format it will be used in. The [README](../../README.md) links here from
its command list.



The steering half of Ster reads hidden states, fits directions from them,
scores and compares those directions, and adds them during generation. Every
operation on a steering artifact is a verb of `ster vector`; generating and
exporting states sit beside it:

```text
ster vector train --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
                  [--revision <REVISION>] [--device cpu] [--layers all]
                  [--method caa|pca|logistic] [--chat-template auto|off]
                  [--precision f32|f16|bf16]
ster vector optimize --model <MODEL> --pairs <PAIRS> --output <OUTPUT> --holdout <FRACTION>
                     [--revision <REVISION>] [--device cpu] [--layers all]
                     [--chat-template auto|off] [--precision f32|f16|bf16]
ster vector curve --model <MODEL> --pairs <PAIRS> --method caa|pca|logistic
                  --holdout <FRACTION> --sizes <N,N,...> [--layers all]
                  [--revision <REVISION>] [--device cpu]
                  [--chat-template auto|off] [--precision f32|f16|bf16]
ster vector evaluate --model <MODEL> --pairs <PAIRS> --vector <VECTOR>
                     [--strengths <S,S,...> --batch-size <N> --max-sequence <TOKENS>]
                     [--revision <REVISION>] [--device cpu]
                     [--chat-template auto|off] [--precision f32|f16|bf16]
ster vector inspect <ARTIFACT>
ster vector compare <ARTIFACT> <ARTIFACT>... --clusters <N> [--layers <L,L,...>]
ster vector project --model <MODEL> --pairs <PAIRS> --vector <VECTOR> --layer <L>
                    --strength <S> --components <N> [--revision <REVISION>]
                    [--device cpu] [--chat-template auto|off] [--precision f32|f16|bf16]
ster vector ablate --model <MODEL> --vector <VECTOR> --strength <S> --output <DIR>
                   [--revision <REVISION>]
ster generate --model <MODEL> (--prompt <PROMPT> | --prompts <SET> --output <FILE>)
              [--vector <VECTOR> --strength <S>]... [--system <FILE>]
              [--adapter <ADAPTER>] [--revision <REVISION>] [--device cpu]
              [--chat-template auto|off] [--precision f32|f16|bf16]
              --max-new-tokens <N> --temperature <T>
              [--top-p <TOP_P>] --seed <SEED>
ster extract --model <MODEL> --input <INPUT> --output <OUTPUT>
             [--revision <REVISION>] [--device cpu] [--layers all]
             [--chat-template auto|off] [--precision f32|f16|bf16]
```

`--layers` takes `all`, a comma list, or a half-open range such as `8..16`,
exactly as it does under `ster tune`, and `--method` names the estimator
`vector train` fits: contrastive activation addition, the leading principal
direction, or a logistic probe. `--chat-template` and `--precision` are on
every one of these commands except `vector inspect` and `vector compare`,
which load no model. Each command prints a pretty JSON document on stdout,
and each is also an operation of `ster request`: `ster request vector/train`,
`vector/optimize`, `vector/curve`, `vector/evaluate`, `vector/inspect`, `vector/compare`, `vector/project`, `vector/ablate`,
`generate` and `extract` read the request body as one JSON document on stdin,
where every flag above is a camelCase field, `chatTemplate` and `precision`
included, each defaulting to what the CLI defaults to, and print NDJSON log
events and one result event carrying the same document. The numbers have no
default on either side: `--holdout`, `--max-new-tokens`, `--temperature` and
`--seed` (and `holdout`, `maxNewTokens`, `temperature`, `seed`) are required,
and leaving one out is refused naming it. `--strength` scales the `--vector`
in the same place and exists only with one: every `--vector` needs its own,
and a strength without a vector is refused. The request carries the same
pairs as `steering: [{"vector": …, "strength": …}]`, each strength required.
`--top-p` is optional; without it generation samples the whole
distribution, and a temperature of zero is argmax. The process ends with
the operation; see [desktop requests](desktop-requests.md).

Every command that reads hidden states counts what it read, so a run with
too few pairs or activations fails with the counts instead of writing a
quieter artifact. A pair set with no pairs is refused on load as
`pair set {path} contains no pairs` (and `pairs import --count N` over an
export that yields fewer is refused as `{source} yields {found} pairs, fewer
than --count N`); a prompt set with no prompts as `prompt set contains no
prompts`. Each forward pass must return one non-empty vector at every
requested layer, or the run stops with `the forward pass returned activations
at {found} of {requested} requested layers (missing [...], not requested
[...])` or `the forward pass returned an empty activation vector at layer
{layer}`; fitting a direction refuses unequal sides as `training requires the
same non-zero number of positive and negative examples; got {positive}
positive and {negative} negative`. `vector train`, `optimize`, `curve`,
`evaluate`, `project` and `extract` all read through the same check, and as
`ster request` operations they end with the same refusal as their error event.

Several `--vector`/`--strength` pairs steer several traits at once, as
wisent's `multi-steer` did: at every layer any of them carries, the sum of
each vector times its strength is added to the residual stream. A direction
and its negation at equal strengths cancel to the unsteered answer, and one
vector named twice at a strength steers as it does once at twice that
strength; every artifact passes the checks a single one does, and strengths
that are all zero are refused, since such a mix would steer nothing.

`ster generate --prompts <SET> --output <FILE>` answers a whole prompt set
(`{"prompts": ["…"]}`, as [`docs/examples/prompts.json`](../examples/prompts.json)) with
the model loaded once and writes one `{"prompt", "model_output"}` per prompt, in the
set's order, to `--output`, then prints its path; every other flag means what it means
for one prompt. `--prompt` and `--prompts` exclude each other, `--output` goes with
`--prompts` only, and an empty set is refused with `prompt set contains no prompts`
before any weight is mapped. A steered sweep is one run per strength:

```bash
ster model fixture create --from docs/examples/model/fixture.json --output .build/prompt-fixture
ster generate --model .build/prompt-fixture --prompts docs/examples/prompts.json --output answers.json \
  --max-new-tokens 16 --temperature 0 --seed 7
```

## Selection

`ster vector optimize` fits every layer-and-method combination on part of the pair
set, ranks the candidates on pairs none of them were fitted on, and writes the
winner. What is new is that it publishes the ranking. It used to print the
choice — layer 9, method pca — which is a result with no evidence attached, and
a chooser that publishes only its choice is asking to be trusted.

The document is the artifact summary plus a `selection` object holding
`holdout`, with `fraction`, `fit_pairs` and `holdout_pairs`, and `candidates`, one row per
layer and method carrying `layer`, `method`, `holdout_accuracy`,
`holdout_margin`, `holdout_effect_size` and `selected`. Exactly one row has
`selected` true. The rows
stay in the order the search walked them rather than sorted by score, so two
runs over the same layers diff line for line. The scores cost nothing to carry:
they were computed to make the decision.

`holdout_effect_size` is Cohen's d of the held-out sides along the
candidate's direction: the mean positive projection minus the mean negative
one, over their spread pooled around each side's own mean, `null` when the
projections do not spread at all. It is the answer wisent's
`check-linearity` gave — whether a trait lies along a line at a layer — as a
number at every layer and method instead of a LINEAR or WEAKLY_LINEAR label
cut at thresholds nobody stated; the caller reads it against whatever bar
their use needs.

The split is the caller's: `--holdout` is the fraction of the pairs held out,
rounded to whole pairs, and what decides whether the ranking means anything is
the two counts it produced. The run says them before it starts —
`fitting each candidate on 3 pairs and ranking on a 1-pair holdout` — and when
the holdout comes out at a single pair it says what that costs:

```text
a one-pair holdout scores every candidate 0 or 1, so this ranking separates almost nothing; add pairs to make the choice mean something
```

That is a fact about the input rather than a defect, so it is stated rather
than refused. A fraction that leaves either side without a pair is refused
before any activation is read:
`a held-out fraction of 0.1 over 4 contrastive pairs leaves 4 pairs to fit and 0 to rank on; both need at least one`.
A fraction outside (0, 1) is refused as
`the held-out fraction must be above zero and below one`. Ranking prefers accuracy and breaks ties on margin,
so on a holdout of one pair the tiebreak is doing all of the work.

The published direction is then refitted on every pair, holdout included, and
the artifact's `metadata` records that in one sentence:
`chosen over 66 candidates on a 1-pair holdout, then refitted on all 4 pairs`.
The split existed to rank candidates, and once the ranking is done, throwing
away the held-out evidence would be paying for the measurement twice. It also
means the `train_accuracy` and `train_margin` the artifact carries are the
refit's numbers over the whole set rather than the holdout scores in the table:
the table is the evidence for the choice, and the artifact's own numbers
describe the direction that was written.

## Pair count

`ster vector curve` answers how many pairs a direction needs, the question
wisent's `optimize-sample-size` asked. The last `--holdout` fraction of the
set is held out exactly as `vector optimize` holds it out; `--method` is then
fitted at every layer on the first `N` pairs of the rest for every `N` in
`--sizes`, and every fit is scored on the same held-out pairs. The answer
lists one point per size and layer (`size`, `layer`, `holdout_accuracy`,
`holdout_margin`, `holdout_effect_size`), the `holdout` it was cut with, and per layer under
`enough` the smallest size that reached the best held-out accuracy any size
reached there. Ster states no size, method or fraction. A size larger than
the pairs left to fit on is refused as
`a size of <N> pairs does not fit in the <FIT> pairs left to fit on after holding out <HELD>`,
and the fraction is refused exactly as `vector optimize` refuses it.

wisent's `optimize-classification` ranked probes per layer; `vector optimize`
ranks the logistic probe beside caa and pca at every layer on the same
holdout. Its `optimize` ran every search at once; here that is `vector
optimize` and `vector curve` run one after the other. Its
`optimization-cache` kept those runs' results; here each run's result is the
artifact or report it writes.

wisent's analysis commands are these primitives composed. Its
`cluster-benchmarks` grouped benchmarks by how alike their directions are:
fit one artifact per benchmark's pairs with `vector train` and group them
with `vector compare --clusters`. Its `geometry-search` looked for one
direction good across benchmarks: `pairs merge` the benchmarks' sets and
`vector optimize` the result, whose holdout spans all of them. Its `zwiad`
report — layer sensitivity, method comparison, similarity between traits —
is `vector optimize`'s table (every layer and method on one holdout), `vector
compare` between traits, and `vector project` for the picture of one.

## Strength

A direction says which way to move the residual stream; how far is a separate
question, and `ster generate` refuses to guess it. `ster vector evaluate --strengths`
answers it on pairs. Each strength named is added with the artifact to the
frozen model, and every pair is scored twice: the steered model's
log-probability of each side minus the unsteered model's. A pair is ordered
when steering raised its positive side more than its negative side, and its
shift is the difference. A strength that pushes too hard drags both sides down
together and orders fewer pairs — the over-steering a representation score
cannot see.

```bash
ster model fixture create --from docs/examples/model/fixture.json --output .build/strength-fixture
ster vector train --model .build/strength-fixture --pairs docs/examples/pairs.json --output calm.ster.json --layers 2
ster vector evaluate --model .build/strength-fixture --pairs docs/examples/pairs.json --vector calm.ster.json \
  --strengths -1,0.5,1,2,4 --batch-size 4 --max-sequence 256
```

The report gains a `strength` object: `method`, `layers`, `pairs`,
`scored_pairs`, `skipped_long`, `selected_strength`, and `candidates`, one row
per strength in the order given, carrying `strength`, `ordered` (the share of
scored pairs ordered), `mean_shift` and `selected`. The choice follows
`optimize`'s rule: the largest share ordered, the larger mean shift breaking a
tie. Score it on pairs the artifact was not fitted on when the choice has to
mean something beyond them.

Ster names no strength, batch or sequence limit. `--batch-size` (pairs per
forward pass) and `--max-sequence` (the longest side measured, in tokens) are
required with `--strengths` and refused without it; a pair longer than the
limit is skipped, counted in `skipped_long` and named on the progress stream.
The refusals: `--strengths needs --batch-size; Ster assumes none`,
`--strengths needs --max-sequence; Ster assumes none`,
`strength selection needs finite strengths, not inf`,
`strength selection requires batch size of at least one`, and the artifact
checks generation makes (another model, another width, a layer the model
lacks). Through `ster request vector/evaluate` the fields are `strengths`,
`batchSize` and `maxSequence`, with the same requirements.

## Inspection

`ster vector inspect` validates an artifact and prints a summary of it. It used to
serialize the artifact itself, which on a twenty-two-layer 2048-wide checkpoint
is forty-five thousand floats down a terminal, while `ster tune inspect` beside
it printed tensor names and shapes. A steering vector's content is not readable
and its shape and length are, so `inspect` now prints the same document `train`
and `optimize` print: `schema_version`, `product`, `model`, `model_revision`,
`trait_name`, `method`, `hidden_size`, `precision`, `chat_template`,
`metadata`, and a `layers` array carrying, per layer, `layer`, `width`, `norm`,
`train_accuracy` and `train_margin`. Nothing was removed from the artifact; the
numbers are still on disk for anything that wants them.

`norm` is the Euclidean length of the direction, accumulated in `f64` because a
two-thousand-term sum of squares in `f32` loses its low bits. Every direction
Ster writes is unit-normalized, so a norm that is not 1.0 to within rounding is
the fastest available sign that a file was written by something other than
Ster.

## Geometry and provenance

- [Comparison, projection and writing directions into weights](steering/geometry.md)
  covers artifact geometry, result fields and useful refusals.
- [Provenance and chat-format transfer](steering/provenance.md) covers native
  compatibility warnings and the published format-transfer measurement.

## Real qualification

[Steering qualification](steering/qualification.md) documents the permanent
prompt-batch, mixture and pair/activation-count journeys, explicit artifact
and experiment inputs, meaningful refusals, retained evidence and GUI boundary.
