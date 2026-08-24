# Contrastive pair

A contrastive pair is one positive prompt and one negative prompt that differ
in the trait being studied; a pair set is a named list of them. Pairs are the
only supervision Ster ever takes: directions are learned from the difference
between the two sides' [activations](activation.md), and
[evaluation](evaluation.md) asks whether each positive out-projects its own
negative.

## Shape

One JSON document (`PairSet` in `src/artifact.rs`):

```json
{
  "trait_name": "calm",
  "pairs": [
    {
      "positive": "the sea is calm and quiet tonight .",
      "negative": "the storm waves crash loud against the rocks ."
    }
  ]
}
```

- `trait_name` — optional (defaults to `""`); copied verbatim into every
  artifact trained from the set. Name the trait, not the experiment.
- `pairs[]` — objects with exactly `positive` and `negative` strings
  (`ContrastivePair`). Order matters twice: pairs are matched positionally
  during training, and `ster optimize` splits the list 80/20 **in file
  order** — shuffle before saving if the file is sorted by anything
  meaningful.

## Lifecycle

`ster train`, `ster optimize`, and `ster evaluate` all take `--pairs` and load
the set the same way. Each prompt of each pair is run through the model
independently and its final-token activation captured per selected layer
(`reading pair 3/8` progress on stderr — one pair is two forward passes).
The set itself is never written back or transformed; it is input only.

## Invariants and refusals

Checked at load (`PairSet::load`), before any model work:

- The file must exist and be valid JSON:
  `failed to read pair set <path>` / `invalid pair set JSON in <path>`.
- At least one pair: `pair set <path> contains no pairs`.
- No side may be empty or whitespace-only:
  `pair set <path> contains an empty positive or negative prompt`.

Checked later, per prompt, by the runtime: every prompt must tokenize to at
least one token (`prompt must not be empty`, `tokenizer produced no tokens`)
and fit the model context. `ster optimize` additionally refuses sets smaller
than four: `optimization requires at least four contrastive pairs`.

## Writing pairs that work

The learned direction is exactly the regularity that separates the two
columns. Keep everything except the trait matched — topic, length, format —
or the direction will encode the confound instead of the trait. The README's
example trait (`truthful`) contrasts evidence-hedged answers with confident
fabrications over the same questions; this corpus's executed toy trait
(`calm`) contrasts calm and stormy sentences over the same scenery. Training
accuracy near 0.5 means the pairs do not separate at that layer —
see [evaluation](evaluation.md).
