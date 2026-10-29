# Steering geometry

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

## Projection

`ster vector project` shows where an artifact moves the model's states as
numbers a caller can plot or check. Every side of every pair is read at
`--layer` twice, unsteered and with the artifact added at `--strength`, and
both reads are projected onto the first `--components` principal components
of the unsteered reads; Ster chooses neither number. A read at a layer is
taken before that layer's own vector is added, so the movement it shows comes
from the vectors at the layers before it. The answer lists every point
(`pair`, `side`, `steered`, `coordinates`), the share of the unsteered variance
each component carries (`explained`, fewer entries when the reads span fewer
directions), the mean movement along the artifact's own unit direction at that
layer (`shift_along_direction`, `null` when the artifact carries no vector
there), the mean distance of the negative sides from the mean positive side
before and after steering, and the share of negatives steering brought nearer.
Refusals: a layer the model lacks, no components, an artifact of another model
or width, and reads that do not vary at all.

## Writing a direction into the weights

`ster vector ablate` makes a direction permanent by directional projection.
At every layer the artifact carries, the two projections that write the
residual stream — the attention output and the feed-forward down projection —
become `W - strength * v (v^T W)` for the layer's unit direction `v`: at
strength one the layer can no longer write along the direction, a smaller
strength removes that share of it, and a negative one amplifies it. Ster
states no strength. The output is an ordinary checkpoint directory written by
the same code `tune merge` writes with — the rewritten weights beside the
source's own config, tokenizer and chat template — so every loader reads it,
and no decoder is built to write it. The answer names every rewritten tensor,
the copied ones and the files written. Refusals: a zero or non-finite
strength, an artifact fitted on another model or width, a layer the model
lacks, a GGUF checkpoint, and a write stored fused with another projection.
