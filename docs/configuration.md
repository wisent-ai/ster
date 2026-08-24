# Configuration

Ster 0.12 has no application configuration file. Runtime behavior is fully
selected by CLI arguments or the equivalent `ster serve` request fields, plus
compile-time Cargo features and two Candle CPU thread environment variables.
There is no precedence stack: omitted fields take the defaults below.

## CLI keys

These are all runtime CLI keys. `--help` and `--version` are clap controls, not
Ster configuration.

| Command | Key | Type | Default | Contract |
|---|---|---:|---|---|
| train, optimize, evaluate, generate, extract | `--model` | string | required | Existing local directory or Hugging Face model id. |
| same | `--revision` | string | `main` for Hub, none for local | Hub ref; local metadata only. |
| same | `--device` | enum | `cpu` | `cpu`, `metal`, or `cuda`. |
| train | `--pairs` | path | required | Pair-set JSON. |
| train | `--output` | path | required | Steering artifact destination. |
| train | `--layers` | layer expression | `all` | Single indices, comma lists, or half-open ranges. |
| train | `--method` | enum | `caa` | `caa`, `pca`, `logistic`; aliases `mean-difference`, `probe`. |
| optimize | `--pairs` | path | required | Pair-set JSON with at least four pairs. |
| optimize | `--output` | path | required | Winning artifact destination. |
| optimize | `--layers` | layer expression | `all` | Candidate layer set. |
| evaluate | `--pairs` | path | required | Evaluation pair-set JSON. |
| evaluate | `--vector` | path | required | Steering artifact. |
| generate | `--prompt` | string | required | Nonblank prompt shorter than context. |
| generate | `--vector` | path | none | Omit for baseline generation. |
| generate | `--strength` | f64 | `1.0` | Signed shared steering scale. |
| generate | `--max-new-tokens` | usize | `128` | Must be greater than zero. |
| generate | `--temperature` | f64 | `0.0` | `<= 0` means argmax; positive means sampling. |
| generate | `--top-p` | f64 | none | Used only with positive temperature. |
| generate | `--seed` | u64 | `42` | Used by sampling; argmax is deterministic. |
| extract | `--input` | path | required | JSON shaped `{"prompts":["..."]}`. |
| extract | `--output` | path | required | Activation artifact destination. |
| extract | `--layers` | layer expression | `all` | Layers to export. |
| inspect | positional `ARTIFACT` | path | required | Steering artifact to validate/print. |
| serve | `--port` | u16 | `0` | Loopback port; `0` asks the OS for an ephemeral port. |

Exact command behavior: [CLI reference](cli.md). Layer grammar:
[layer selection](concepts/layer-selection.md).

Ster passes temperature/top-p values to Candle without an additional range
check. Invalid sampling parameters may therefore surface as a Candle error;
use `temperature > 0` and `0 < top_p <= 1`. Strength is not clipped.

## Serve request keys

Request JSON uses camelCase. Unknown JSON fields are ignored by serde. Missing
fields are defaulted before required-field validation. All path values are
server-local and relative paths use the serve process's working directory.

| Endpoint | Key | Type | Default |
|---|---|---:|---|
| train, optimize, evaluate, generate, extract | `model` | string | required |
| same | `revision` | string or null | null (Hub resolver defaults to `main`) |
| same | `device` | string | `cpu` |
| train | `pairs`, `output` | string paths | required |
| train | `layers` | string | `all` |
| train | `method` | string | `caa` |
| optimize | `pairs`, `output` | string paths | required |
| optimize | `layers` | string | `all` |
| evaluate | `pairs`, `vector` | string paths | required |
| generate | `prompt` | string | required |
| generate | `vector` | string or null | null; blank also means baseline |
| generate | `strength` | number | `1.0` |
| generate | `maxNewTokens` | unsigned integer | `128` |
| generate | `temperature` | number | `0.0` |
| generate | `topP` | number or null | null |
| generate | `seed` | unsigned integer | `42` |
| extract | `input`, `output` | string paths | required |
| extract | `layers` | string | `all` |
| inspect | `artifact` | string path | required |

The protocol, response envelopes, and serialization behavior are in the
[serve API reference](serve-api.md).

## Build-time features

| Cargo feature | Default | Effect |
|---|---|---|
| `metal` | off | Enables Candle Metal and makes `--device metal` select device 0. |
| `cuda` | off | Enables Candle CUDA and makes `--device cuda` select device 0. |

The default feature set is empty. Devices never auto-detect and there is no
runtime fallback: a CPU-only binary refuses `metal`/`cuda`. All backends use
F32 in Ster.

## Environment and local files

Ster source reads no environment variables directly. Its Candle dependency
honors these CPU thread knobs:

| Variable | Valid value | Fallback |
|---|---|---|
| `RAYON_NUM_THREADS` | positive integer | physical CPU/P-core count, at least 1 |
| `CANDLE_NUM_THREADS` | positive integer | physical CPU/P-core count, at least 1 |

`RAYON_NUM_THREADS` sizes Candle's Rayon pool and CPU matmul parallelism;
`CANDLE_NUM_THREADS` sizes its barrier pool. Missing, non-integer, zero, or
negative values silently use the hardware fallback.

Hub resolution calls hf-hub `Api::new`, not its environment-aware builder.
Consequently `HF_HOME` and `HF_ENDPOINT` do not change Ster 0.12. The default
client uses:

- cache: `~/.cache/huggingface/hub`;
- optional bearer token file: `~/.cache/huggingface/token`.

There is no Ster environment key for model, device, revision, cache, endpoint,
token, offline mode, output directory, log level, or serve port.

## Model config keys

Ster does not own a separate model-config schema. It requires top-level
`model_type: "llama"`, then deserializes the whole `config.json` through
Candle 0.11's `LlamaConfig`. Dimensions, heads, rope settings, EOS ids, context
length, norm epsilon, and tied embeddings therefore come from the checkpoint,
not Ster flags. Required files and resolution rules are documented in
[model resolution](concepts/model-resolution.md).

## Repository release metadata

`.wisent-release.json` configures repository delivery, not the installed
binary. Its complete present key tree is: `schema_version`, `product`,
`releases`; `version_source.{kind,path,pattern}`;
`platforms.<name>.{runner_platform,quality,build.argv,stage}`; and
`promotion.{channels,reconcile}`, `inputs`, `deliveries`. It declares
`darwin-arm64` and `linux-amd64`, both built by
`sh scripts/build-release.sh`, promoted on `stable`, with no quality commands,
inputs, or deliveries. Changing it cannot configure a running Ster process.
