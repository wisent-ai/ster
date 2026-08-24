# Walkthrough: train, select, inspect, extract, and steer

This is the recorded offline Ster 0.12 session behind the quick start. It ran
on CPU against the seeded toy Llama checkpoint produced by
[`examples/make-toy-model.py`](examples/make-toy-model.py). No network, GPU, or
model service was used. Random weights make the generated text meaningless;
the session demonstrates the real data path and gates.

## Prepare the inputs

```bash
WORK=$(mktemp -d)
cd "$WORK"
python3 <ster-repo>/docs/examples/make-toy-model.py toy-llama
cp <ster-repo>/docs/examples/pairs.json .
cp <ster-repo>/docs/examples/prompts.json .
STER=<ster-repo>/target/release/ster
```

The executed pair file has eight matched calm/stormy pairs. The checkpoint has
four layers (`0..3`) and hidden width 64.

## Train two layers

```console
$ $STER train --model toy-llama --pairs pairs.json --layers 1..3 --method caa --output calm.ster.json
reading pair 1/8
reading pair 2/8
reading pair 3/8
reading pair 4/8
reading pair 5/8
reading pair 6/8
reading pair 7/8
reading pair 8/8
{
  "artifact": {
    "hidden_size": 64,
    "layers": [
      {
        "layer": 1,
        "train_accuracy": 0.875,
        "train_margin": 0.5598376393318176
      },
      {
        "layer": 2,
        "train_accuracy": 0.875,
        "train_margin": 0.7613550424575806
      }
    ],
    "method": "caa",
    "model": "toy-llama",
    "model_revision": null,
    "product": "ster",
    "schema_version": 1,
    "trait_name": "calm"
  }
}
```

The eight `reading pair` lines were stderr; the JSON summary was stdout. The
compact `calm.ster.json` includes both full 64-value vectors, not just the
summary.

## Search method and layer

```console
$ $STER optimize --model toy-llama --pairs pairs.json --layers all --output calm.best.ster.json
reading pair 1/8
reading pair 2/8
reading pair 3/8
reading pair 4/8
reading pair 5/8
reading pair 6/8
reading pair 7/8
reading pair 8/8
```

The command evaluated `caa`, `pca`, and `logistic` on every layer using the
ordered 80/20 holdout, then retrained the winner on all eight pairs. The
recorded output artifact selected CAA at layer 3:

```json
{
  "model": "toy-llama",
  "trait_name": "calm",
  "method": "caa",
  "hidden_size": 64,
  "vectors": [
    { "layer": 3, "train_margin": 0.92032135, "train_accuracy": 0.875 }
  ],
  "metadata": { "selection": "80/20 holdout over method and layer" }
}
```

That excerpt omits schema/product/revision and the 64 values; it reports exact
fields read from `calm.best.ster.json`. The saved scores are the retrained
full-set scores, not the holdout scores used for selection.

For comparison, separately executed layer-2 artifacts scored CAA
`0.875 / 0.76135504`, PCA `0.625 / 0.23738393`, and logistic
`1.0 / 0.73598194` as training accuracy/margin. Method quality is
layer-dependent; a probe's perfect training score is not proof it generalizes.

## Evaluate the two-layer artifact

```console
$ $STER evaluate --model toy-llama --pairs pairs.json --vector calm.ster.json
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

This deliberately reuses the training set to show mechanics. Use a different
held-out pair file for evidence.

## Compare generation

```console
$ $STER generate --model toy-llama --prompt "describe the evening lake ." --max-new-tokens 12
loud sea hard hard , white , white , white loud loud

$ $STER generate --model toy-llama --vector calm.ster.json --strength 1.0 --prompt "describe the evening lake ." --max-new-tokens 12
white white white howls howls howls drifts , white white white white
```

Both runs used deterministic argmax. The changed text proves the real
intervention reached logits. It says nothing semantic about a random model.
The same experiment at strength `2.0` collapsed to one repeated token; larger
positive values reached EOS immediately. Negative values need attached CLI
syntax, for example `--strength=-1.0`.

## Export raw activations

```console
$ $STER extract --model toy-llama --input prompts.json --layers 0,3 --output activations.json
extracting prompt 1/2
extracting prompt 2/2
activations.json
```

The recorded file has schema 1, product `ster`, model `toy-llama`, width 64,
two records, and two full vectors per record under string keys `"0"` and
`"3"`. No Ster command consumes this activation artifact; it is the external
analysis boundary.

## Inspect and trip a gate

```bash
$STER inspect calm.best.ster.json
```

`inspect` pretty-printed the complete schema-1 artifact without loading a
model. A copy whose `product` was changed to `other` was refused:

```text
Error: failed to inspect tampered.json

Caused by:
    artifact belongs to product "other", not Ster
```

The same product gate applies before evaluation or intervention. Other
operator-facing failures and fixes are in the [runbook](runbook.md).
