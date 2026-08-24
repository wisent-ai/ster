# CLI reference

One binary, seven subcommands. Verified against `ster 0.12.0` (`src/main.rs`);
every usage block below is pasted from the built binary's `--help`.

```text
Usage: ster <COMMAND>

Commands:
  train     Train steering vectors from positive and negative prompts
  optimize  Select the best method and layer on an 80/20 holdout
  evaluate  Measure pair ordering for a steering artifact
  generate  Generate text with an optional steering artifact
  extract   Export hidden representations for arbitrary prompts
  inspect   Print and validate a Ster steering artifact
  serve     Loopback HTTP/JSON backend for desktop apps
```

Conventions that hold everywhere:

- **Exit code** is `0` on success, non-zero on any refusal; the refusal is one
  `Error: <sentence>` on stderr ([runbook](runbook.md) lists them all).
- **stdout** carries exactly the command's result document (JSON for
  `train`/`optimize`/`evaluate`/`inspect`, generated text for `generate`, the
  output path for `extract`, the ready line for `serve`). Progress lines
  (`reading pair 3/8`, `extracting prompt 1/2`) go to **stderr**.
- Negative numeric values must be attached with `=` (`--strength=-2.0`);
  clap otherwise reads `-2.0` as a flag and exits 2 with
  `error: unexpected argument '-2' found`.

## Shared model arguments

`train`, `optimize`, `evaluate`, `generate`, and `extract` all load a model
first, with the same three arguments (`ModelArgs`):

| Argument | Default | Meaning |
|---|---|---|
| `--model <MODEL>` | required | Hugging Face model id, or a local directory containing `config.json`, `tokenizer.json`, and Safetensors weights. A path that exists as a directory is used in place; anything else is treated as a Hub id and fetched. |
| `--revision <REVISION>` | `main` (Hub) / none (local) | Immutable Hub revision (branch, tag, or commit SHA). For Hub models the *resolved* commit SHA is recorded in artifacts regardless; for local directories the value is recorded verbatim if given. |
| `--device <DEVICE>` | `cpu` | `cpu`, `metal`, or `cuda`. The GPU values only work in binaries built with the matching feature; otherwise: `this Ster binary was built without the metal feature`. |

Loading refuses any checkpoint whose `config.json` is not
`model_type: "llama"` — see [model resolution](concepts/model-resolution.md).

## ster train

```text
Usage: ster train [OPTIONS] --model <MODEL> --pairs <PAIRS> --output <OUTPUT>

      --pairs <PAIRS>        JSON file with trait_name and contrastive pairs
      --output <OUTPUT>      Output Ster steering artifact
      --layers <LAYERS>      Comma-separated layers, half-open ranges such as 8..16, or all [default: all]
      --method <METHOD>      Direction training method: caa, pca, or logistic [default: caa]
```

Captures both activations per pair per selected layer, trains one
unit-normalized direction per layer with `--method`, self-scores each on the
training pairs, and writes one [steering artifact](concepts/steering-artifact.md)
containing **all** selected layers. Stdout is a summary (scores, no vectors):

```json
{ "artifact": { "hidden_size": 64, "layers": [ { "layer": 1,
  "train_accuracy": 0.875, "train_margin": 0.5598376393318176 } ],
  "method": "caa", "model": "toy-llama", "model_revision": null,
  "product": "ster", "schema_version": 1, "trait_name": "calm" } }
```

`--method` accepts the aliases `mean-difference` (= `caa`) and `probe`
(= `logistic`). Layer grammar: [layer selection](concepts/layer-selection.md).

## ster optimize

```text
Usage: ster optimize [OPTIONS] --model <MODEL> --pairs <PAIRS> --output <OUTPUT>

      --pairs <PAIRS>
      --output <OUTPUT>
      --layers <LAYERS>      [default: all]
```

Grid search over all three methods × every selected layer. Pairs are split
80/20 in file order (`(n*4/5)` train, clamped so both splits are non-empty);
each candidate trains on the first split and is scored on the holdout; the
winner is the highest holdout accuracy, ties broken by margin. The winning
(method, layer) is then **retrained on all pairs** and written as a
single-layer artifact with
`metadata.selection = "80/20 holdout over method and layer"`. Refuses pair
sets smaller than four (`optimization requires at least four contrastive
pairs`). Stdout is the same summary shape as `train`.

## ster evaluate

```text
Usage: ster evaluate [OPTIONS] --model <MODEL> --pairs <PAIRS> --vector <VECTOR>
```

Loads an existing artifact, re-captures activations for `--pairs` at exactly
the artifact's layers, and reports per-layer pair-ordering accuracy and mean
margin ([evaluation](concepts/evaluation.md)). Refuses an artifact whose
`model` string differs from `--model` (`artifact model "toy-llama" does not
match runtime model "toy-llama-copy"`). Stdout:

```json
{ "model": "toy-llama", "trait_name": "calm", "method": "caa",
  "pair_count": 8, "layers": [ { "layer": 1, "accuracy": 0.875,
  "margin": 0.55983764 } ] }
```

## ster generate

```text
Usage: ster generate [OPTIONS] --model <MODEL> --prompt <PROMPT>

      --prompt <PROMPT>
      --vector <VECTOR>
      --strength <STRENGTH>              [default: 1]
      --max-new-tokens <MAX_NEW_TOKENS>  [default: 128]
      --temperature <TEMPERATURE>        Zero selects deterministic argmax generation [default: 0]
      --top-p <TOP_P>
      --seed <SEED>                      [default: 42]
```

Autoregressive generation with a KV cache; stdout is the generated text only
(prompt excluded, special tokens skipped). Without `--vector` it is a plain
baseline. With `--vector`, the artifact is gated (schema, product, model
string, hidden width, layer range, finiteness) and then applied as an
[intervention](concepts/intervention.md): `strength × vector` added to the
residual stream after each artifact layer, every step.

Sampling: `--temperature 0` (the default) is deterministic argmax and
ignores `--top-p` and `--seed`; a positive temperature samples the full
distribution, or the top-p nucleus when `--top-p` is set. Generation stops at
`--max-new-tokens`, at any of the model's EOS token ids, or at the model's
context limit, whichever comes first — an immediate EOS yields an empty
stdout line. Prompts must be non-empty and shorter than the model's
`max_position_embeddings`.

## ster extract

```text
Usage: ster extract [OPTIONS] --model <MODEL> --input <INPUT> --output <OUTPUT>

      --input <INPUT>        JSON file shaped as {"prompts": ["..."]}
      --output <OUTPUT>
      --layers <LAYERS>      [default: all]
```

Exports final-token [activations](concepts/activation.md) for every prompt at
every selected layer into one JSON activation artifact (`schema_version` 1,
`product` `"ster"`, model identity, `hidden_size`, then one record per prompt
with a `layers` map). Stdout is the output path; progress
(`extracting prompt 1/2`) is on stderr.

## ster inspect

```text
Usage: ster inspect <ARTIFACT>
```

Loads a steering artifact through the same validation used everywhere else,
then pretty-prints the whole document (vectors included) to stdout. No model
is loaded — this is the offline check that a file is a well-formed Ster
artifact. Failures wrap the refusal: `Error: failed to inspect
tampered.json` … `Caused by: artifact belongs to product "other", not Ster`.

## ster serve

```text
Usage: ster serve [OPTIONS]

      --port <PORT>  Port to bind; 0 selects an ephemeral port [default: 0]
```

Binds 127.0.0.1, prints exactly one ready line to stdout —
`{"port":61058,"ready":true}` — and then serves HTTP until killed. Every
endpoint reuses the CLI's own code paths. Full surface:
[serve-api](serve-api.md).
