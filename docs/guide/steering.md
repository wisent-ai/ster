# Steering

Choosing a direction, reading what it is, and what it costs to fit one out of
the format it will be used in. The [README](../../README.md) links here from
its command list.



The steering half of Ster reads hidden states, fits directions from them,
scores those directions, and adds them during generation. Six commands:

```text
ster train --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
           [--revision <REVISION>] [--device cpu] [--layers all]
           [--method caa|pca|logistic] [--chat-template auto|off]
           [--precision f32|f16|bf16]
ster optimize --model <MODEL> --pairs <PAIRS> --output <OUTPUT>
              [--revision <REVISION>] [--device cpu] [--layers all]
              [--chat-template auto|off] [--precision f32|f16|bf16]
ster evaluate --model <MODEL> --pairs <PAIRS> --vector <VECTOR>
              [--revision <REVISION>] [--device cpu]
              [--chat-template auto|off] [--precision f32|f16|bf16]
ster generate --model <MODEL> --prompt <PROMPT> [--vector <VECTOR>]
              [--adapter <ADAPTER>] [--revision <REVISION>] [--device cpu]
              [--chat-template auto|off] [--precision f32|f16|bf16]
              [--strength 1.0] [--max-new-tokens 128] [--temperature 0.0]
              [--top-p <TOP_P>] [--seed 42]
ster extract --model <MODEL> --input <INPUT> --output <OUTPUT>
             [--revision <REVISION>] [--device cpu] [--layers all]
             [--chat-template auto|off] [--precision f32|f16|bf16]
ster inspect <ARTIFACT>
```

`--layers` takes `all`, a comma list, or a half-open range such as `8..16`,
exactly as it does under `ster tune`, and `--method` names the estimator
`train` fits: contrastive activation addition, the leading principal
direction, or a logistic probe. `--chat-template` and `--precision` are on
every one of these commands except `inspect`, which loads no model. Each
command prints a pretty JSON document on stdout, and each is also an operation
of `ster request`: `ster request train`, `ster request optimize`,
`ster request evaluate`, `ster request generate`, `ster request extract` and
`ster request inspect` read the request body as one JSON document on stdin,
where every flag above is a camelCase field, `chatTemplate` and `precision`
included, each defaulting to what the CLI defaults to, and print NDJSON log
events and one result event carrying the same document. The process ends with
the operation; see [desktop requests](desktop-requests.md).

## Selection

`ster optimize` fits every layer-and-method combination on part of the pair
set, ranks the candidates on pairs none of them were fitted on, and writes the
winner. What is new is that it publishes the ranking. It used to print the
choice — layer 9, method pca — which is a result with no evidence attached, and
a chooser that publishes only its choice is asking to be trusted.

The document is the artifact summary plus a `selection` object holding
`holdout`, with `fit_pairs` and `holdout_pairs`, and `candidates`, one row per
layer and method carrying `layer`, `method`, `holdout_accuracy`,
`holdout_margin` and `selected`. Exactly one row has `selected` true. The rows
stay in the order the search walked them rather than sorted by score, so two
runs over the same layers diff line for line. The scores cost nothing to carry:
they were computed to make the decision.

The split is reported rather than assumed, because "80/20" is a ratio and what
decides whether the ranking means anything is the two counts it produced. The
run says them before it starts —
`fitting each candidate on 3 pairs and ranking on a 1-pair holdout` — and when
the holdout comes out at a single pair it says what that costs:

```text
a one-pair holdout scores every candidate 0 or 1, so this ranking separates almost nothing; add pairs to make the choice mean something
```

That is a fact about the input rather than a defect, so it is stated rather
than refused: four pairs is the smallest set `optimize` accepts, and four pairs
yield a one-pair holdout. Ranking prefers accuracy and breaks ties on margin,
so on a holdout of one pair the tiebreak is doing all of the work.

The published direction is then refitted on every pair, holdout included, and
the artifact's `metadata` records that in one sentence:
`chosen over 66 candidates on a 1-pair holdout, then refitted on all 4 pairs`.
The split existed to rank candidates, and once the ranking is done, throwing
away a fifth of the evidence would be paying for the measurement twice. It also
means the `train_accuracy` and `train_margin` the artifact carries are the
refit's numbers over the whole set rather than the holdout scores in the table:
the table is the evidence for the choice, and the artifact's own numbers
describe the direction that was written.

## Inspection

`ster inspect` validates an artifact and prints a summary of it. It used to
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

## Provenance

A steering artifact records the precision and the chat-template decision of the
run that fitted it, and `ster evaluate` and `ster generate` check them against
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
and margin `ster evaluate` reports:

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

