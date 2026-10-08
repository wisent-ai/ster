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
ster vector evaluate --model <MODEL> --pairs <PAIRS> --vector <VECTOR>
                     [--strengths <S,S,...> --batch-size <N> --max-sequence <TOKENS>]
                     [--revision <REVISION>] [--device cpu]
                     [--chat-template auto|off] [--precision f32|f16|bf16]
ster vector inspect <ARTIFACT>
ster vector compare <ARTIFACT> <ARTIFACT>... --clusters <N> [--layers <L,L,...>]
ster generate --model <MODEL> (--prompt <PROMPT> | --prompts <SET> --output <FILE>)
              [--vector <VECTOR> --strength <S>] [--system <FILE>]
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
`vector/optimize`, `vector/evaluate`, `vector/inspect`, `vector/compare`,
`generate` and `extract` read the request body as one JSON document on stdin,
where every flag above is a camelCase field, `chatTemplate` and `precision`
included, each defaulting to what the CLI defaults to, and print NDJSON log
events and one result event carrying the same document. The numbers have no
default on either side: `--holdout`, `--max-new-tokens`, `--temperature` and
`--seed` (and `holdout`, `maxNewTokens`, `temperature`, `seed`) are required,
and leaving one out is refused naming it. `--strength` (`strength`) scales a
steering vector and exists only with one: required with `--vector`, refused
without it. `--top-p` is optional; without it generation samples the whole
distribution, and a temperature of zero is argmax. The process ends with
the operation; see [desktop requests](desktop-requests.md).

`ster generate --prompts <SET> --output <FILE>` answers a whole prompt set
(`{"prompts": ["…"]}`, as [`docs/examples/prompts.json`](../examples/prompts.json)) with
the model loaded once and writes one `{"prompt", "model_output"}` per prompt, in the
set's order, to `--output`, then prints its path; every other flag means what it means
for one prompt. `--prompt` and `--prompts` exclude each other, `--output` goes with
`--prompts` only, and an empty set is refused with `prompt set contains no prompts`
before any weight is mapped. A steered sweep is one run per strength:

```bash
ster toy-model toy-model
ster generate --model toy-model --prompts docs/examples/prompts.json --output answers.json \
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
`holdout_margin` and `selected`. Exactly one row has `selected` true. The rows
stay in the order the search walked them rather than sorted by score, so two
runs over the same layers diff line for line. The scores cost nothing to carry:
they were computed to make the decision.

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
ster toy-model toy-model
ster vector train --model toy-model --pairs docs/examples/pairs.json --output calm.ster.json --layers 2
ster vector evaluate --model toy-model --pairs docs/examples/pairs.json --vector calm.ster.json \
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

## Comparison

`ster vector compare A B [C ...] --clusters <N>` compares steering artifacts
fitted for different traits on one model. It loads no model: the directions
are read from the artifacts. At every layer all of them carry (or every layer
`--layers` names, a comma list each artifact must carry) it reports each
pair's cosine similarity, then their mean over those layers as `similarity`
and the most similar and most different pair. `uniqueness` is, per artifact,
the length of the part of its directions that the other artifacts'
directions at the same layers do not span, over their whole length: one for
a direction nothing else explains, none for one the others already contain.
`distance` is the mean distance between unit directions, taking whichever of
a pair's two signs lies nearer, because a direction and its negation pick
out one axis; average linkage on it cuts the artifacts into `--clusters`
groups (`clusters`, one group number per artifact in argument order), and
`silhouette` is that cut's mean silhouette over the artifacts that share a
group, or `null` when none does. Ster chooses no group count. Refusals:
fewer than two artifacts, artifacts of different models or widths, more
groups than artifacts, a named layer an artifact does not carry, no layer
every artifact carries, and a direction with no usable length.

## Provenance

A steering artifact records the precision and the chat-template decision of the
run that fitted it, and `ster vector evaluate` and `ster generate` check them against
the run that is consuming it. Both call the same helper the tune half has used
for adapters, so a direction read in a space it was not fitted in says so on
the progress stream:

```text
warning: this direction was trained with chat template off and this run encodes applied, so the number below describes a format it was not trained in
warning: this direction was trained at precision f32 and this run maps the base weights at f16, so it is being read in a different space than it was fitted in
```

A warning and not a refusal, deliberately. Both mismatches are things an
operator may want on purpose — measuring how far a direction transfers out of
the format it was fitted in is a real question, and the measurement below is
exactly that experiment — and a refusal would make it impossible rather than
merely deliberate. Only an unnoticed mismatch is a defect. An artifact written
before these fields existed carries `null` in both and warns about neither: an
absent record is not a disagreement.

## What fitting out of format costs

[Chat templates](#chat-templates) covers the flag itself and its three
outcomes. The steering half has one further consequence, and it is the sharper
one: the hidden-state read behind `train`, `optimize`, `evaluate` and `extract`
encodes every prompt through the template, so a direction is fitted in the
space it will be applied in. That was not true before this release — pair text
went to the model raw while `generate` rendered its prompt through the template
— and the difference is not a rounding question. A direction is a displacement
between two points in the residual stream, and where those points sit depends
on the markers around the text that produced them: under a template the model
is answering a user turn, without one it is continuing a document.

Measured on `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, at `--precision f32`, over
all 22 layers, with the four-pair set at `~/.stado/work/loop/pairs.json` for
the trait `calm and measured, never alarmed`. One direction was fitted with
`--chat-template off` and another with `auto`, and each was evaluated under
both. The two columns are the mean over the 22 layers of the per-layer accuracy
and margin `ster vector evaluate` reports:

| fitted | evaluated | mean accuracy | mean margin |
| --- | --- | --- | --- |
| `off` | `off` | 1.0000 | +4.9451 |
| `auto` | `auto` | 1.0000 | +1.0917 |
| `off` | `auto` | 0.7841 | +0.0168 |
| `auto` | `off` | 0.7614 | +0.1160 |

The two matched runs separate all four pairs at every one of the 22 layers. The
two crossed runs do not: accuracy falls below 1.0 at 12 of the 22 layers for
off evaluated as auto, the mean lands near 0.78 whichever way the crossing
runs, and the margin collapses by roughly two orders of magnitude. Layer 21 is
the clearest single reading: +20.6697 fitted and evaluated `off`, and -0.0486
for that same direction evaluated `auto`.

The sign is the part worth stopping on. In the three deepest layers — 19, 20
and 21, in both crossings — the crossed margin is negative, which is not a weak
direction but a wrong one: the projection orders the two sides of a pair
backwards, so adding that direction during generation pushes toward the side
the operator labelled negative. A direction fitted out of format does not
merely lose resolution in the layers where steering is usually applied. It
points the other way.

That is why the read encodes through the template by default rather than
offering it as something to remember. The two matched rows also show what the
flag is not: `off` and `auto` each separate the set perfectly in their own
space, and the larger raw margin of the `off` run is a property of raw-text
geometry rather than a better direction. A margin is only comparable with
another margin taken in the same format, which is exactly what the artifact now
records so that `evaluate` can say when it is not.

