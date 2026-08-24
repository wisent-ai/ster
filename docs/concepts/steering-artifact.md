# Steering artifact

A steering artifact is Ster's durable, model-bound collection of one or more
[layer directions](direction.md). `train` and `optimize` write it; `inspect`,
`evaluate`, and `generate` read it. The file is compact JSON, not Safetensors,
and carries enough identity and shape information to refuse a plausible vector
on the wrong model.

## Schema 1

`SteeringArtifact` in `src/artifact.rs` has this complete wire shape:

```json
{
  "schema_version": 1,
  "product": "ster",
  "model": "toy-llama",
  "model_revision": null,
  "trait_name": "calm",
  "method": "caa",
  "hidden_size": 64,
  "vectors": [
    {
      "layer": 2,
      "values": [0.14712493, 0.16871634, -0.061927322],
      "train_margin": 0.76135504,
      "train_accuracy": 0.875
    }
  ],
  "metadata": {}
}
```

The displayed `values` array is shortened; the executed file has 64 values.
Every field is required on read except `metadata`, which defaults to `{}` for
schema-1 files that omit it.

| Field | Contract |
|---|---|
| `schema_version` | Must equal the build's `ARTIFACT_SCHEMA_VERSION`, currently `1`. |
| `product` | Must be exactly `"ster"`. |
| `model` | The exact `--model` string used for training; runtime matching is string equality. |
| `model_revision` | Resolved Hub commit SHA; for local directories, the supplied revision or `null`. |
| `trait_name` | Copied verbatim from the pair set; may be empty. |
| `method` | Canonical `caa`, `pca`, or `logistic`. |
| `hidden_size` | Model residual width; must be non-zero and match every vector and the runtime model. |
| `vectors` | Non-empty array of layer vectors. `train` keeps every selected layer; `optimize` keeps one. |
| `metadata` | String-to-string map. `optimize` adds `selection: "80/20 holdout over method and layer"`; `train` leaves it empty. |

A `LayerVector` contains a 0-based `layer`, exactly `hidden_size` finite F32
`values`, and the training-set `train_margin` and `train_accuracy`. Artifact
validation does not independently bound the scores or reject duplicate layer
numbers; the commands create well-formed, sorted vectors, and generation's
`BTreeMap` plan would let the last duplicate replace an earlier one. Do not
hand-author duplicates.

## Lifecycle

- `SteeringArtifact::new` stamps schema 1 and product `ster`.
- `save` validates, creates parent directories, serializes compact JSON, and
  writes the file.
- `load` reads, deserializes, and validates.
- `ster inspect` is the model-free CLI for the same load-and-validate path and
  then pretty-prints the complete document.
- `evaluate` adds an exact artifact-model-string gate before scoring.
- `generate` adds model string, model width, and model layer-count gates before
  constructing an [intervention](intervention.md).

## Refusals

Validation is fail-closed:

- `artifact schema <N> is unsupported; this Ster build reads schema 1`
- `artifact belongs to product "<PRODUCT>", not Ster`
- `artifact has no steering vectors`
- `layer <L> has vector width <ACTUAL>, expected <HIDDEN_SIZE>`
- `layer <L> contains a non-finite value`

I/O adds `failed to read steering artifact <PATH>`, `invalid steering artifact
JSON in <PATH>`, `failed to create <PARENT>`, and `failed to write steering
artifact <PATH>`. Runtime mismatch sentences are in the
[runbook](../runbook.md).

Schema 1 is versioned but Ster is pre-1.0: do not infer forward compatibility.
Keep the original model revision and pair set beside a release artifact when
reproducibility matters.
