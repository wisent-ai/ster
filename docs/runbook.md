# Runbook

Ster fails closed: a CLI refusal exits non-zero and `anyhow` prints `Error:`
plus its context chain. In serve mode, failures before a job starts are 400/404
JSON; failures after a 200 NDJSON stream starts become a stderr log and a final
`status: 1` result. Match the stable source-authored sentence below; substituted
values are shown as `<PLACEHOLDERS>`.

## Model and checkpoint loading

| Sentence | Meaning | Fix |
|---|---|---|
| `unknown device "<VALUE>"; expected cpu, metal, or cuda` | `--device`/`device` is not one of the three lowercase names. | Use a supported name. |
| `this Ster binary was built without the metal feature` | Metal was selected on a CPU-only build. | Rebuild/install with `--features metal`, then retry. |
| `this Ster binary was built without the cuda feature` | CUDA was selected without CUDA support. | Rebuild/install with `--features cuda` on a supported host. |
| `failed to initialize Metal device` | Feature exists but Candle could not open Metal device 0. | Check OS/device support and driver/runtime state; CPU is the explicit fallback. |
| `failed to initialize CUDA device` | Feature exists but Candle could not open CUDA device 0. | Check CUDA device/driver/runtime compatibility; CPU is the explicit fallback. |
| `failed to list <DIR>` | A local model directory could not be enumerated. | Check existence and read/execute permissions. |
| `model config is missing: <PATH>` | Local `config.json` is absent or not a regular file. | Restore it at the model root. |
| `tokenizer is missing: <PATH>` | Local `tokenizer.json` is absent or not a regular file. | Restore it at the model root. |
| `model directory contains no safetensors weights` | No eligible top-level `.safetensors` file was found. | Add model weights; optimizer-named files do not count. |
| `failed to initialize Hugging Face Hub client` | hf-hub could not build its client/cache/token configuration. | Check home/cache permissions and TLS/client setup, or use a local directory. |
| `failed to read model repository <MODEL>` | Hub repository metadata lookup failed. | Check model id, revision, network, authentication, and gated access. |
| `model <MODEL> publishes no safetensors weights` | Repository siblings contain no eligible weights. | Choose a Safetensors model or materialize a supported local directory. |
| `failed to download <WEIGHT_NAME>` | A discovered shard could not be downloaded. | Check access/network/cache space; retry the same immutable revision. |
| `failed to read <CONFIG_PATH>` | Resolved config could not be read. | Check file/cache integrity and permissions. |
| `invalid model config <CONFIG_PATH>` | `config.json` is not JSON. | Replace it with the checkpoint's valid config. |
| `model architecture "<TYPE>" is unsupported by this Ster build; use a Hugging Face Llama-family checkpoint with model_type=llama` | `model_type` is missing/non-string (shown as `""`) or not `llama`. | Use a Llama-family checkpoint whose config says exactly `llama`. |
| `invalid Llama config <CONFIG_PATH>` | JSON passed the architecture gate but not Candle's Llama schema. | Use a complete compatible Llama config; do not hand-delete required dimensions. |
| `failed to load tokenizer <PATH>: <DETAIL>` | tokenizer JSON or its model is invalid/incompatible. | Restore the tokenizer shipped with the checkpoint. |
| `failed to map <N> model weight files` | Safetensors mapping/header/tensor loading failed. | Check every shard, permissions, completeness, disk/memory, and config/weight agreement. |

A model-looking string that is not an existing directory is treated as a Hub
id. If an intended local path produces Hub errors, fix or create the directory.
Hub `config.json`/`tokenizer.json` fetches and low-level Candle operations can
also surface dependency-generated detail without an additional Ster sentence;
retain the full `Error:` chain when reporting them.

## Pair and prompt inputs

| Sentence | Meaning | Fix |
|---|---|---|
| `failed to read pair set <PATH>` | Pair file cannot be opened. | Correct the path and permissions. |
| `invalid pair set JSON in <PATH>` | JSON does not deserialize to `trait_name` plus `pairs[].positive/negative`. | Fix syntax, field names, and string types. |
| `pair set <PATH> contains no pairs` | `pairs` is empty. | Add at least one matched pair. |
| `pair set <PATH> contains an empty positive or negative prompt` | A side is blank after trimming. | Fill both sides of every pair. |
| `optimization requires at least four contrastive pairs` | Holdout selection has fewer than four pairs. | Provide at least four; use `train` when selection is not needed. |
| `failed to read <PATH>` | `extract` cannot read its prompt file. | Correct path/permissions. |
| `invalid prompt JSON in <PATH>` | Extract input is not `{"prompts":[strings...]}`. | Fix syntax and shape. |
| `prompt set contains no prompts` | Extract `prompts` is empty. | Add at least one prompt. |
| `prompt must not be empty` | A generation, pair, or extract prompt is blank. | Supply non-whitespace text. |
| `failed to tokenize prompt: <DETAIL>` | Tokenizer rejected the text. | Check tokenizer compatibility/input and retain the detail. |
| `tokenizer produced no tokens` | Tokenization succeeded with an empty id sequence. | Supply text recognized by a valid tokenizer. |
| `prompt contains <N> tokens, model context allows fewer than <LIMIT>` | Generation prompt already fills/exceeds context. | Shorten it; prompt length must be strictly below the limit. |
| `failed to decode generated tokens: <DETAIL>` | Tokenizer could not decode the generated suffix. | Check tokenizer/model vocabulary agreement and preserve the detail. |
| `max_new_tokens must be greater than zero` | Generation token budget is 0. | Set `--max-new-tokens`/`maxNewTokens` to at least 1. |

An immediate EOS is not an error: generation succeeds with an empty text line
or `{"text":""}`.

## Layer selection

| Sentence | Meaning | Fix |
|---|---|---|
| `model has no layers` | Loaded config reports zero decoder blocks. | Use a valid model config. |
| `invalid layer "<SEGMENT>"` | A non-range segment is not an unsigned integer. | Use `5`, a valid range, or `all`. |
| `invalid layer range "<SEGMENT>"` | A range endpoint is not an unsigned integer. | Use syntax such as `8..16`. |
| `layer range "<SEGMENT>" must have start < end` | Range is empty or reversed. | Make the half-open end greater than start. |
| `no layers selected` | The expression contains only commas/whitespace. | Select at least one index or `all`. |
| `at least one layer is required` | A library call or artifact supplies an empty layer list. | Provide a non-empty list/artifact. |
| `layer <L> is outside the model's 0..<MAX> range` | Selection/artifact layer is too large. | Select only 0-based decoder layers in range. |

## Training and numerical routines

These normally indicate bad direct library inputs; CLI capture creates
consistent rows.

| Sentence | Meaning | Fix |
|---|---|---|
| `unknown training method "<VALUE>"; expected caa, pca, or logistic` | Method is unsupported. | Use a canonical method or documented alias. |
| `training requires the same non-zero number of positive and negative examples` | Row sides are empty or counts differ. | Supply matched non-empty rows. |
| `activation vectors are empty` | Hidden width is zero. | Supply model-width rows. |
| `activation vectors must have one finite, consistent width` | A row width differs or any value is NaN/infinite. | Normalize shape and remove non-finite inputs. |
| `training produced a zero or non-finite direction` | Learned vector cannot be unit-normalized. | Improve non-degenerate contrastive data. |
| `direction width <D> does not match activation width <A>` | Evaluation direction has wrong width. | Use a direction from the same model space. |
| `cosine similarity requires equal non-empty vectors` | Direct similarity inputs differ in width or are empty. | Supply equal non-empty vectors. |
| `cosine similarity is undefined for a zero vector` | Either similarity input has near-zero norm. | Use non-zero vectors. |

## Artifact I/O and validation

| Sentence | Meaning | Fix |
|---|---|---|
| `failed to read steering artifact <PATH>` | File cannot be opened. | Correct path/permissions. |
| `invalid steering artifact JSON in <PATH>` | JSON syntax or required field/type is wrong. | Regenerate or repair against schema 1. |
| `artifact schema <N> is unsupported; this Ster build reads schema 1` | Schema version differs. | Use a schema-1 artifact with this build; do not relabel incompatible data. |
| `artifact belongs to product "<PRODUCT>", not Ster` | Product discriminator is not `ster`. | Supply a genuine Ster artifact. |
| `artifact has no steering vectors` | Width is 0 or vector list empty. | Retrain/regenerate the artifact. |
| `layer <L> has vector width <W>, expected <H>` | Artifact's own width invariant fails. | Restore untampered output or retrain. |
| `layer <L> contains a non-finite value` | A vector contains NaN/infinity. | Retrain; do not apply it. |
| `failed to create <PARENT>` | Artifact parent directory creation failed. | Fix path/permissions. |
| `failed to write steering artifact <PATH>` | Artifact serialization reached a filesystem write failure. | Fix space, permissions, or destination. |
| `failed to inspect <PATH>` | `inspect` context wrapper; the next cause is the actual I/O/schema refusal. | Fix the deepest cause. |
| `artifact model "<A>" does not match runtime model "<B>"` | `evaluate` requires exact model-string equality. | Load the artifact's exact model string or retrain. |
| `artifact was trained for model "<A>", current model is "<B>"` | `generate` model-string gate failed. | Same: exact string match or retrain. |
| `artifact width <A> does not match model width <M>` | Runtime hidden width differs. | Use the training checkpoint/artifact pair. |
| `layer <L> steering vector width <W> does not match model width <H>` | Low-level plan width gate failed. | Supply model-width vectors. |
| `steering plan contains no vectors` | Direct low-level plan construction got no entries. | Supply at least one vector. |
| `failed to apply steering at layer <L>` | Candle rejected broadcast-add during the forward pass. | Treat as model/plan shape or device failure; retain the chained detail. |

Activation export may return `failed to write <PATH>` for its destination;
fix parent permissions/space. Unlike artifact save, export's parent-directory
creation may surface a raw OS error without a Ster wrapper.

## CLI parser errors and exit behavior

Missing required arguments, invalid numeric values, unknown flags/subcommands,
and values outside clap's target type fail before Ster code runs. Clap prints a
lowercase `error:` usage message and exits 2. In particular, attach a negative
number: `--strength=-2.0`; `--strength -2.0` is parsed as an unexpected flag
(the captured build reported `error: unexpected argument '-2' found`).

After parsing, any `anyhow::Result` failure exits non-zero and begins with
`Error:`. Progress already written to stderr is not rolled back, and output
files are not transactional: if a filesystem fails during a write, inspect or
remove the partial destination before retrying.

## Serve transport and request validation

### Listener and HTTP parser

| Sentence | HTTP/log | Fix |
|---|---|---|
| `failed to bind the serve port` | Process exits before ready line. | Choose a free port; verify local socket policy. |
| `serve accept failed: <DETAIL>` | Stderr; listener continues. | Diagnose transient OS/socket exhaustion from detail. |
| `malformed request line` | 400 JSON. | Send a normal `METHOD /path HTTP/1.1` line. |
| `request body too large` | 400 JSON; limit is 1 MiB. | Reduce request JSON; pass large data by server-local file path. |
| `request body is not valid JSON` | 400 JSON. | Send valid JSON (or blank for `{}`). |
| `unknown endpoint: <METHOD> <PATH>` | 404 JSON. | Use an exact method/path under `/v1`. Query/fragment is ignored for routing. |

An invalid or absent `Content-Length` is treated as zero, not chunked input;
this minimal server does not implement chunked request bodies.

### Required request fields

These are all pre-job validation sentences, returned as 400 JSON in this order:

- `train requires a model`
- `train requires a pairs file`
- `train requires an output path`
- `optimize requires a model`
- `optimize requires a pairs file`
- `optimize requires an output path`
- `evaluate requires a model`
- `evaluate requires a pairs file`
- `evaluate requires a steering artifact`
- `generate requires a model`
- `generate requires a prompt`
- `extract requires a model`
- `extract requires a prompt input file`
- `extract requires an output path`
- `inspect requires a steering artifact`

Once streaming starts, the HTTP status remains 200 even when a workflow fails.
The final event then has `status: 1` and `json.error`; do not gate success on
HTTP alone. A successful job ends with exactly one `status: 0` result.
