# Quick start

Every command below was executed on a checkout of this repository (Ster
0.12.0, macOS arm64, CPU). The walkthrough uses a tiny offline toy checkpoint
so it needs no download, no GPU, and no Hugging Face account. The toy model's
weights are seeded random numbers — its text is deterministic gibberish, and
that is the point: the mechanics (reading, training, gating, steering) are
identical to a real model's, just instant. To do the same on a real model,
substitute `--model meta-llama/Llama-3.2-1B` (or any local Llama-family
Safetensors directory) everywhere `toy-llama` appears.

## 1. Install

From this checkout:

```bash
cargo install --path . --locked
```

Or from GitHub: `cargo install --git https://github.com/wisent-ai/ster
--locked`. Add `--features metal` or `--features cuda` for GPU backends —
they are compile-time choices, not flags ([configuration](configuration.md)).

```console
$ ster --version
ster 0.12.0
```

## 2. Make an offline toy model

```bash
WORK=$(mktemp -d)
cd "$WORK"
python3 <ster-repo>/docs/examples/make-toy-model.py toy-llama
```

This writes `config.json` (`model_type: "llama"`, 4 layers, hidden width 64),
a WordLevel `tokenizer.json`, and `model.safetensors` — the exact three
inputs [model resolution](concepts/model-resolution.md) requires from a local
directory.

## 3. Define the trait as contrast

A trait is a [contrastive pair set](concepts/contrastive-pair.md): matched
prompts that differ only in the trait. Save as `pairs.json` (this corpus
trains a toy "calm vs stormy" trait; the toy tokenizer only knows its own
~60 words, so pairs stick to that vocabulary):

```json
{
  "trait_name": "calm",
  "pairs": [
    {
      "positive": "the sea is calm and quiet tonight .",
      "negative": "the storm waves crash loud against the rocks ."
    },
    {
      "positive": "the water lies still in the harbor .",
      "negative": "the wind howls hard over the dark hills ."
    }
  ]
}
```

The full executed set has 8 pairs and is checked in as
[`examples/pairs.json`](examples/pairs.json) — `ster optimize` refuses fewer
than 4.

## 4. Train

```console
$ ster train --model toy-llama --pairs pairs.json --layers 1..3 --method caa --output calm.ster.json
reading pair 1/8
...
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

Progress goes to stderr; the summary JSON goes to stdout; the full
[steering artifact](concepts/steering-artifact.md) (including the vectors)
is in `calm.ster.json`. `--layers 1..3` is a half-open range: layers 1 and 2
([layer selection](concepts/layer-selection.md)).

## 5. Evaluate

```console
$ ster evaluate --model toy-llama --pairs pairs.json --vector calm.ster.json
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

Accuracy is the fraction of pairs whose positive prompt projects higher onto
the direction than its negative; margin is the mean projection gap
([evaluation](concepts/evaluation.md)). Evaluate on held-out pairs, not the
training set, when the number matters.

## 6. Generate — baseline, then steered

```console
$ ster generate --model toy-llama --prompt "describe the evening lake ." --max-new-tokens 12
loud sea hard hard , white , white , white loud loud

$ ster generate --model toy-llama --vector calm.ster.json --strength 1.0 \
    --prompt "describe the evening lake ." --max-new-tokens 12
white white white howls howls howls drifts , white white white white
```

Same prompt, same seed — the added direction visibly changes the token
distribution (on a real model, it changes the trait; on this toy it changes
gibberish). Strength is linear and signed; pass negative values as
`--strength=-2.0` (with `=`, or the CLI parses `-2.0` as a flag). Large
strengths dominate the stream entirely — at `--strength 2.0` the toy model
emits one token forever, and past that it can hit end-of-sequence immediately
and return an empty line. Calibrate: [intervention](concepts/intervention.md).

## 7. Where to go next

- The full command surface, argument by argument: [cli](cli.md).
- Method-and-layer selection with `ster optimize`, artifact gating, and
  activation export, all executed:
  [walkthrough-steering](walkthrough-steering.md).
- The same engine over loopback HTTP for desktop apps:
  [walkthrough-serve](walkthrough-serve.md).
- Every refusal sentence you might hit on the way: [runbook](runbook.md).
