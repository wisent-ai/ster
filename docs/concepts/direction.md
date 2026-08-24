# Direction (steering vector)

A direction is a unit-length `f32` vector in activation space — one per
layer — that points from "negative" toward "positive" for a trait. It is
Ster's core object: reading projects onto it, steering adds along it. In
artifacts it appears as a `LayerVector`; the README and CLI call the whole
per-layer collection a *steering vector*.

## Shape

Inside a [steering artifact](steering-artifact.md) (`LayerVector` in
`src/artifact.rs`):

```json
{
  "layer": 3,
  "values": [0.10141639, 0.16717353, -0.03978117, ...],
  "train_margin": 0.92032135,
  "train_accuracy": 0.875
}
```

- `layer` — the 0-based block index the vector was read from and will be
  added to.
- `values` — exactly `hidden_size` finite floats.
- `train_accuracy`, `train_margin` — the [evaluation](evaluation.md) scores
  measured at save time, kept with the vector so an artifact is
  self-describing.

## Lifecycle

1. **Trained** — `train_direction` (`src/representation.rs`) turns matched
   positive/negative activation rows into a raw direction using one
   [training method](training-method.md).
2. **Normalized** — every method's output is L2-normalized before it leaves
   training. This is unconditional; a direction whose norm is zero or
   non-finite is refused (`training produced a zero or non-finite
   direction`). Because directions are unit-length, `--strength` is the
   entire magnitude knob and strengths are comparable across methods.
3. **Scored and saved** — accuracy/margin are computed on the training pairs
   and stored beside the values.
4. **Applied** — at generation, the vector is scaled by `strength` and added
   to the residual stream at its layer ([intervention](intervention.md)).

## Sign convention

Positive projection = more of the trait, because training subtracts negative
from positive (`caa` averages `pos − neg`; `pca` sign-aligns its principal
component with that mean difference; `logistic` labels positives 1). So
`--strength 1.0` pushes generation *toward* the trait and `--strength=-1.0`
pushes away. Keep the convention in the pair file, not in your head: if the
columns are swapped, the direction is simply negated.

## Invariants and refusals

Enforced by artifact validation and the runtime gate:

- Width must match the artifact's `hidden_size`:
  `layer 1 has vector width 10, expected 64`.
- Every component finite: `layer 3 contains a non-finite value`.
- Width must match the loaded model at apply time:
  `artifact width 64 does not match model width 2048` and, at plan
  construction, `layer 3 steering vector width 10 does not match model
  width 64`.
- The layer must exist in the loaded model:
  `layer 4 is outside the model's 0..3 range`.

Directions are comparable with `ster::representation::cosine_similarity`
(library API): it refuses mismatched or empty inputs (`cosine similarity
requires equal non-empty vectors`) and zero vectors (`cosine similarity is
undefined for a zero vector`).
