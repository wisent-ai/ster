# What is Ster

Ster is Wisent's native Rust toolkit for representation reading and activation
steering in open-weight Llama-family language models. One binary and one
library crate (`ster` 0.12.0, Candle runtime, no Python) do the whole loop:
read hidden states out of a transformer, learn a direction that separates a
trait, judge whether the direction actually separates it, and add that
direction back into the residual stream while the model generates. The whole
product is three moving parts.

## Reading: activations out

Ster owns its own Llama decoder loop instead of using a generic inference API,
because the loop needs one exact read: after each selected transformer block,
the final-token residual state is copied out as an `f32` vector of
`hidden_size` width. That vector is the [activation](concepts/activation.md)
— the only representation Ster ever trains on. `ster extract` exposes the
read directly, exporting activations for arbitrary prompts as a versioned
JSON document; `ster train`, `ster optimize`, and `ster evaluate` use the same
read internally, once per prompt per selected layer. Models load from a local
directory (`config.json`, `tokenizer.json`, Safetensors weights) or from the
Hugging Face Hub, always in F32, on CPU by default. Anything whose
`config.json` does not say `model_type: "llama"` is refused before weights
are loaded. Details: [model resolution](concepts/model-resolution.md).

## Learning and judging: directions from contrast

A trait is defined operationally, by a
[contrastive pair set](concepts/contrastive-pair.md): matched positive and
negative prompts that differ in the trait. For each selected layer, Ster
captures both sides' activations and trains a unit-length
[direction](concepts/direction.md) with one of three
[methods](concepts/training-method.md) — `caa` (mean difference), `pca`
(principal component of pairwise differences), or `logistic` (a linear
probe). Every direction is immediately judged by the same two numbers:
pair-ordering **accuracy** (fraction of pairs where the positive projects
higher than the negative) and mean projection **margin**
([evaluation](concepts/evaluation.md)). `ster optimize` runs the full grid —
every method at every selected layer — on an 80/20 holdout and keeps the
single best layer. The result is a
[steering artifact](concepts/steering-artifact.md): a versioned JSON document
that records the model identity, resolved revision, trait, method, hidden
width, and the per-layer normalized vectors with their training scores.

## Intervening: directions back in

`ster generate` runs normal autoregressive generation, or steered generation
when given an artifact: at each layer that has a vector, `strength × vector`
is added to the residual stream after that block, on every forward step
([intervention](concepts/intervention.md)). Before anything is added, the
artifact is gated: wrong schema, wrong product, wrong model string, wrong
vector width, out-of-range layer, or a non-finite value each produce a
one-sentence refusal instead of a wrong-model steering run. The same engine
is reachable over HTTP: `ster serve` is a loopback (127.0.0.1) JSON backend
for desktop apps, streaming each job's progress and result as NDJSON while
reusing exactly the CLI's code paths ([serve API](serve-api.md)).

## What Ster is not

Ster controls local open-weight models only. Hosted model routing belongs to
Brama; fleet placement, credentials, and release delivery belong to Stado and
Skarbiec. The current runtime accepts `model_type: "llama"` and fails closed
on everything else. The earlier Python package and `wisent` command were
removed in the Rust cutover; `pip install ster` installs unrelated software
from another publisher. Ster keeps no daemon state, no database, and no
configuration file — its only durable outputs are the JSON documents it
writes where you point it. Boundaries: [architecture](architecture.md).

## The first three commands

```bash
ster train --model meta-llama/Llama-3.2-1B --pairs pairs.json \
  --layers 12..20 --method caa --output truthful.ster.json
```

One direction per selected layer, trained from contrastive pairs, saved as a
steering artifact; a training-score summary prints on stdout.

```bash
ster inspect truthful.ster.json
```

Validate and print an artifact — schema, product, model, width, and every
vector — without loading any model.

```bash
ster generate --model meta-llama/Llama-3.2-1B \
  --vector truthful.ster.json --strength 1.0 \
  --prompt "Explain the result and cite only evidence you can verify."
```

Steered generation. Omit `--vector` for the unsteered baseline of the same
prompt. The end-to-end path on a no-download toy checkpoint is
[quick-start](quick-start.md).

## The rest of the corpus

- **Nouns** — [contrastive pair](concepts/contrastive-pair.md),
  [activation](concepts/activation.md),
  [direction](concepts/direction.md),
  [training method](concepts/training-method.md),
  [layer selection](concepts/layer-selection.md),
  [evaluation](concepts/evaluation.md),
  [steering artifact](concepts/steering-artifact.md),
  [intervention](concepts/intervention.md),
  [model resolution](concepts/model-resolution.md).
- **Reference** — the full [CLI](cli.md), [Rust library API](library-api.md),
  and loopback [serve API](serve-api.md).
- **Executed end to end** — [training and steering](walkthrough-steering.md),
  [a serve session](walkthrough-serve.md), runnable
  [examples](examples/README.md).
- **When it refuses** — every error sentence with meaning and fix:
  [runbook](runbook.md).
- **Boundaries and knobs** — [architecture](architecture.md),
  [configuration](configuration.md).
