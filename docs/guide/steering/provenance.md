# Steering provenance

A steering artifact records the precision and the chat-template decision of the
run that fitted it, and `ster vector evaluate` and `ster generate` check them against
the run that is consuming it. Both call the same helper the tune half uses for
adapters, so a direction read in a space it was not fitted in says so on the
progress stream:

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

[Chat templates](../tuning/chat-templates.md) covers the flag itself and its
outcomes. The hidden-state read behind `train`, `optimize`, `evaluate` and
`extract` encodes every prompt through the template, so a direction is fitted
in the space it will be applied in. A direction is a displacement between two
points in the residual stream, and where those points sit depends on the
markers around the text that produced them: under a template the model is
answering a user turn, without one it is continuing a document.

The [published chat-template comparison](https://ster.wisent.com/docs/concept-chat-template#what-fitting-out-of-format-costs)
measured `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, at `f32`, over all 22 layers,
with a four-pair set for the trait `calm and measured, never alarmed`. One
direction was fitted with `--chat-template off` and another with `auto`, and
each was evaluated under both. The columns are the mean over those layers of
the per-layer accuracy and margin `ster vector evaluate` reports:

| fitted | evaluated | mean accuracy | mean margin |
| --- | --- | --- | --- |
| `off` | `off` | 1.0000 | +4.9451 |
| `auto` | `auto` | 1.0000 | +1.0917 |
| `off` | `auto` | 0.7841 | +0.0168 |
| `auto` | `off` | 0.7614 | +0.1160 |

The two matched runs separate all four pairs at every one of the 22 layers.
The two crossed runs do not: accuracy falls below 1.0 at 12 of the 22 layers
for off evaluated as auto, and the margin collapses by roughly two orders of
magnitude. Layer 21 is the clearest single reading: +20.6697 fitted and
evaluated `off`, and -0.0486 for that same direction evaluated `auto`.

The sign is the part worth stopping on. In the three deepest layers — 19, 20
and 21, in both crossings — the crossed margin is negative, which is not a weak
direction but a wrong one: the projection orders the two sides of a pair
backwards, so adding that direction during generation pushes toward the side
the operator labelled negative. A direction fitted out of format does not
merely lose resolution in the layers where steering is usually applied. It
points the other way.

That is why the read encodes through the template by default rather than
offering it as something to remember. The matched rows also show what the flag
is not: `off` and `auto` each separate the set perfectly in their own space,
and the larger raw margin of the `off` run is a property of raw-text geometry
rather than a better direction. A margin is only comparable with another
margin taken in the same format, which is exactly what the artifact records
so that `evaluate` can say when it is not.
