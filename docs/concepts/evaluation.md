# Evaluation

Evaluation asks whether a trained [direction](direction.md) orders each matched
[contrastive pair](contrastive-pair.md) correctly at one layer. It is the same
calculation used while training, while selecting a method and layer, and by
`ster evaluate`; no separate classifier or generation judge is involved.

## The two measures

For positive activation $p$, negative activation $n$, and direction $d$, Ster
computes the pair margin

$$m=(p\cdot d)-(n\cdot d).$$

A pair is correct only when $m>0$; a zero margin is not correct.

- **accuracy** is `correct_pair_count / pair_count`, in `[0, 1]`;
- **margin** is the arithmetic mean of all pair margins and may be negative.

Directions are unit-normalized before scoring, so the margin is in activation
projection units and is comparable across training methods for the same model
and layer. It is not a probability, confidence interval, or generation-quality
score.

## Three places it appears

1. **`ster train`** scores every selected layer on the same pairs used to train
   it. The values become `vectors[].train_accuracy` and
   `vectors[].train_margin` in the artifact. These are training scores, not
   held-out evidence.
2. **`ster optimize`** splits pairs in file order: the first
   `floor(4n/5)` pairs train each candidate and the remainder score it. The
   highest holdout accuracy wins; exact accuracy ties use the higher margin.
   The winning method and layer are then retrained on all pairs, so the scores
   saved in the final artifact are again full-set training scores rather than
   the holdout scores that selected it.
3. **`ster evaluate`** captures a supplied pair set at every layer carried by
   an artifact and returns fresh scores without changing the artifact.

The executed toy evaluation was:

```json
{
  "model": "toy-llama",
  "trait_name": "calm",
  "method": "caa",
  "pair_count": 8,
  "layers": [
    { "layer": 1, "accuracy": 0.875, "margin": 0.55983764 },
    { "layer": 2, "accuracy": 0.875, "margin": 0.76135504 }
  ]
}
```

## Reading the result

An accuracy near 0.5 means the pair ordering is not reliably separated at that
layer; a high training score may still be memorization or a confound in the
pair design. Keep a held-out pair file for the final `evaluate`, and compare
unsteered and steered generation separately: projection ordering does not
prove that generation changed in the intended way.

## Invariants and refusals

`evaluate_direction` requires equal, non-empty positive and negative rows, one
finite non-zero width, and a direction of that same width. Its source-authored
refusals are:

- `training requires the same non-zero number of positive and negative examples`
- `activation vectors are empty`
- `activation vectors must have one finite, consistent width`
- `direction width <D> does not match activation width <A>`

The CLI additionally gates the pair set, artifact, model identity, and artifact
layers before or during capture; see the [runbook](../runbook.md).
