# Training method

The training method is the rule that turns matched positive/negative
[activation](activation.md) rows into one raw [direction](direction.md).
There are exactly three (`TrainingMethod` in `src/representation.rs`), and
all three end the same way: the result is L2-normalized, so methods differ
only in *where* the unit vector points.

| CLI name | Alias | Enum | Rule |
|---|---|---|---|
| `caa` | `mean-difference` | `Caa` | mean of `positive − negative` over pairs (contrastive activation addition) |
| `pca` | — | `Pca` | first principal component of the centered pairwise differences |
| `logistic` | `probe` | `Logistic` | weight vector of a logistic-regression probe separating positive from negative |

Anything else is refused before the model loads:
`unknown training method "ridge"; expected caa, pca, or logistic`.
Artifacts always record the canonical name (`"caa"`, `"pca"`, `"logistic"`),
never the alias.

## caa — mean difference

The average of per-pair differences: for each pair, subtract the negative
activation from the positive, then average across pairs. Cheap, stable, and
the default. This is the direction the other two methods are compared
against; `pca` is even sign-aligned to it.

## pca — principal difference direction

Computes all pairwise differences, centers them on their mean, and runs 64
power-iteration steps to find the dominant principal component; the result
is sign-flipped if needed so it points the same way as the mean difference.
Degenerate inputs (a zero mean, a collapsed component) fall back to a basis
vector during iteration rather than dividing by zero; a genuinely
inseparable set still fails the final normalization
(`training produced a zero or non-finite direction`).

Use when the trait varies along a consistent axis but individual pairs are
noisy: PCA finds the axis of maximal difference variance rather than the
average offset.

## logistic — linear probe

Full-batch gradient descent on logistic loss: 300 iterations, learning rate
0.1 decayed by `1/(1 + 0.01·step)`, L2 penalty 1e-4, weights initialized to
zero, positives labeled 1 and negatives 0. The learned weight vector (no
bias term) is the direction. This is the classic *probe*: it optimizes
class separation directly, so it can find directions `caa` misses when the
means overlap — on the executed toy set it was the only method to reach
train accuracy 1.0 at layer 2 (`caa` 0.875, `pca` 0.625).

## Choosing — or not choosing

`ster train --method` picks one method for every selected layer.
`ster optimize` refuses to make you choose: it scores all three methods at
every layer on an 80/20 holdout and keeps the single best (method, layer)
pair ([evaluation](evaluation.md)). Method quality is empirical and
layer-dependent; prefer `optimize` whenever the pair set has enough pairs
(at least four, by refusal).

## Shared refusals

All methods validate their input rows first: equal non-zero counts of
positive and negative rows (`training requires the same non-zero number of
positive and negative examples`), non-empty vectors (`activation vectors are
empty`), and one finite consistent width (`activation vectors must have one
finite, consistent width`).
