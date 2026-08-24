# Layer selection

Layer selection names which transformer blocks Ster reads from or trains on.
Layers are 0-based decoder-block indices, `0` through
`num_hidden_layers − 1`; the count comes from the loaded model's
`config.json`, so the same selection string can be valid for one model and
refused for another.

## Grammar

`--layers` (on `train`, `optimize`, `extract`; default `all`) accepts
(`parse_layers` in `src/workflow.rs`):

| Form | Example | Meaning |
|---|---|---|
| `all` | `all` | every layer, `0..count` |
| single index | `5` | layer 5 |
| half-open range | `8..16` | layers 8 through 15 (end excluded) |
| comma list | `3,5,8..12` | union of the segments |

Segments are trimmed, empty segments ignored, and the final list is sorted
and deduplicated — `5,3,3..6` selects `3, 4, 5`. Ranges are **half-open**,
matching Rust: `1..3` is layers 1 and 2, and the README's `12..20` is
layers 12–19.

## Refusals

Parsing and validation fail closed, in this order:

- unparsable segment: `invalid layer "x"` or `invalid layer range "8..x"`;
- reversed or empty range: `layer range "8..8" must have start < end`;
- nothing selected (e.g. `--layers " , "`): `no layers selected`;
- out of range for the loaded model:
  `layer 4 is outside the model's 0..3 range`;
- a model reporting zero layers: `model has no layers`.

The runtime applies the same bound check wherever layer indices arrive from
an artifact instead of a flag (`validate_layers` in `src/runtime.rs`, plus
`at least one layer is required`), so a vector trained on a deeper model is
refused at generation time rather than silently dropped.

## Where selection matters

- **`train`** writes one direction per selected layer into a single
  artifact; wide selections make bigger artifacts (each vector is
  `hidden_size` floats) and cost one capture per layer per prompt.
- **`optimize`** treats the selection as the search space and keeps exactly
  one winning layer.
- **`extract`** exports activations for every selected layer.
- **`evaluate` and `generate`** take no `--layers` flag: the artifact's own
  vectors decide which layers are scored or steered.

Which layers encode a trait is empirical — middle layers are a common
starting point, but `ster optimize --layers all` is the grounded way to find
out, as executed in
[walkthrough-steering](../walkthrough-steering.md).
