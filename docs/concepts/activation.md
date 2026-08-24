# Activation

An activation is the model's residual-stream state for the **final token** of
a prompt, read immediately after one transformer block: an `f32` vector of
`hidden_size` width. It is the single representation everything else in Ster
is built on — directions are trained on activations, evaluation projects
activations, and steering adds to the very stream they are read from.

## Where exactly it is read

Ster owns its Llama decoder loop (`SteeringLlama::forward` in `src/model.rs`)
so the read point is exact and stable:

- after block `i`'s attention and MLP residual additions,
- at the last position of the current sequence,
- converted to F32 (the runtime computes in F32 throughout),
- **before** any steering vector for that same layer is added — capture at
  layer `i` sees the unsteered output of block `i`, even in a steered run.

Layer indices are 0-based block indices, `0..num_hidden_layers`
([layer selection](layer-selection.md)). The embedding output and the final
norm are not capturable; only block outputs are.

## Lifecycle

- `ster train` / `ster optimize` / `ster evaluate` capture activations
  internally — two forward passes per pair — and discard them after training
  or scoring. Nothing is cached between commands; every command re-reads the
  model.
- `ster extract` is the direct export path: arbitrary prompts in
  (`{"prompts": ["..."]}`), one **activation artifact** out.

## The activation artifact

`ster extract` writes one JSON document (`ActivationArtifact` in
`src/workflow.rs`), executed here against the toy checkpoint:

```json
{
  "schema_version": 1,
  "product": "ster",
  "model": "toy-llama",
  "model_revision": null,
  "hidden_size": 64,
  "records": [
    {
      "prompt": "the sea is calm and quiet tonight .",
      "layers": { "0": [0.0711, -0.0236, -0.0592, ...], "3": [...] }
    }
  ]
}
```

`records[].layers` maps layer index (as a JSON string key) to the full
`hidden_size`-wide vector. The document is written once, whole; it is an
export for external analysis, and no Ster command reads it back.

## Invariants and refusals

- Prompts must be non-empty and tokenize to at least one token:
  `prompt must not be empty`, `tokenizer produced no tokens`.
- The prompt set for `extract` must parse and be non-empty:
  `failed to read <path>`, `invalid prompt JSON in <path>`,
  `prompt set contains no prompts`.
- Requested layers must exist: `layer 4 is outside the model's 0..3 range`,
  `at least one layer is required`.
- Downstream training refuses inconsistent captures — activation rows must
  share one finite width (`activation vectors must have one finite,
  consistent width`, `activation vectors are empty`) — which in practice can
  only happen to library callers assembling rows by hand, not to CLI users.
