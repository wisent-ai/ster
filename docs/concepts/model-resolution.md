# Model resolution

Model resolution turns the `--model`, optional `--revision`, and `--device`
inputs into one F32 Candle `Runtime`. Ster supports Hugging Face Llama-family
Safetensors checkpoints from either a local directory or the Hub.

## Local directory

If the `--model` string names an existing directory, Ster never contacts the
network. It requires:

- `config.json`;
- `tokenizer.json`;
- at least one top-level file whose extension is `.safetensors` and whose file
  name does not contain `optimizer`.

Weight names are sorted before memory mapping. A sharded checkpoint works when
all shards are top-level Safetensors files; the index JSON is not consulted.
`--revision`, when supplied for a local model, is metadata only: it is recorded
verbatim in output artifacts and does not select files. Without it,
`model_revision` is `null`.

A nonexistent directory-looking string is not treated as a missing local path:
it falls through and is treated as a Hub repository id. Use an existing path
(or fix the path) to force local resolution.

## Hugging Face Hub

Anything that is not an existing directory is a Hub model id. Ster requests the
repository at `--revision`, defaulting to `main`; downloads `config.json` and
`tokenizer.json`; discovers every published `.safetensors` sibling excluding
names containing `optimizer` or `training_args`; and downloads each weight
file. The repository's resolved commit SHA, not the branch name, becomes
`model_revision` in artifacts.

Ster calls `hf_hub::api::sync::Api::new`, whose dependency version uses
`~/.cache/huggingface/hub` and reads an optional token from
`~/.cache/huggingface/token`. This constructor does **not** use hf-hub's
`from_env` path, so `HF_HOME` and `HF_ENDPOINT` do not configure Ster 0.12.
There is no Ster-specific token or offline flag. For a deterministic offline
run, pass a local directory.

## Architecture gate

The config is first parsed as JSON and its top-level `model_type` must be the
string `"llama"`. Missing, null, or any other value is refused before weights
are mapped:

```text
model architecture "gpt2" is unsupported by this Ster build; use a Hugging Face Llama-family checkpoint with model_type=llama
```

The same bytes must then deserialize as Candle's `LlamaConfig`. Ster builds its
own Llama decoder loop, supports default and Llama-3 rope scaling through
Candle's config types, and always loads and computes weights as F32. Quantized
formats and non-Safetensors weights are not resolution paths.

## Device choice

`cpu` is always available. `metal` and `cuda` are compile-time Cargo features
and select device 0; passing either to a binary built without its feature is a
refusal, not a CPU fallback. Build with `--features metal` or `--features cuda`
and then pass the matching `--device` value.

## Refusals

Local and shared load failures begin with:

- `failed to list <DIR>`
- `model config is missing: <PATH>`
- `tokenizer is missing: <PATH>`
- `model directory contains no safetensors weights`
- `failed to read <CONFIG_PATH>`
- `invalid model config <CONFIG_PATH>`
- `invalid Llama config <CONFIG_PATH>`
- `failed to load tokenizer <PATH>: <DETAIL>`
- `failed to map <N> model weight files`

Hub setup adds:

- `failed to initialize Hugging Face Hub client`
- `failed to read model repository <MODEL>`
- `model <MODEL> publishes no safetensors weights`
- `failed to download <WEIGHT_NAME>`

Device failures and operational fixes are collected in the
[runbook](../runbook.md).
