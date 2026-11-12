# How finding a model and mapping its weights fail

Every declared way loading a model fails at its two ends, whichever command
loads it: finding the checkpoint's files in a local directory or on the
Hugging Face Hub, and mapping its weights into the decoder. Reading its
config is on [model-config.md](model-config.md), adapters and trained
artifacts on [model-load.md](model-load.md). How a refusal reads and which
status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/model.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.model.checkpoint`

find a checkpoint's config, tokenizer and weights in a local directory, or fetch them from a Hugging Face Hub repository into the Hub cache

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.model.checkpoint`: 9 declared cause(s) of its own.

<a id="ster-model-checkpoint-directory-missing"></a>

### `ster.model.checkpoint.directory-missing`

- **What failed:** finding model {model}
- **Cause:** it names a local path (it starts with /, ./, ../ or ~/) and no checkpoint directory is there: {state}
- **Observed state shown:** model: {model}
- **Fix:** name an existing checkpoint directory, or a Hugging Face model id such as org/name; `ster model fixture create` writes a local one
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_local_path_that_is_no_directory_is_refused`

<a id="ster-model-checkpoint-directory-unlistable"></a>

### `ster.model.checkpoint.directory-unlistable`

- **What failed:** listing checkpoint directory {path}
- **Cause:** the directory cannot be read: {error}
- **Observed state shown:** path: {path}
- **Fix:** give this account read access to {path}, or name another checkpoint directory
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::an_unreadable_directory_is_refused`

<a id="ster-model-checkpoint-config-absent"></a>

### `ster.model.checkpoint.config-absent`

- **What failed:** finding the config of checkpoint {place}
- **Cause:** no {file} is there
- **Observed state shown:** path: {path}
- **Fix:** use a checkpoint that holds its config.json (or Mistral's params.json) beside its weights, as `ster model fixture create` writes one
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_directory_without_a_config_is_refused`

<a id="ster-model-checkpoint-tokenizer-absent"></a>

### `ster.model.checkpoint.tokenizer-absent`

- **What failed:** finding the tokenizer of checkpoint {place}
- **Cause:** {place} publishes no tokenizer: none of {files} is there
- **Observed state shown:** looked for: {files}
- **Fix:** use a checkpoint that ships its tokenizer.json, or write one beside the weights with `ster model tokenizer create`
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_directory_without_a_tokenizer_is_refused`

<a id="ster-model-checkpoint-weights-absent"></a>

### `ster.model.checkpoint.weights-absent`

- **What failed:** finding the weights of checkpoint {place}
- **Cause:** {place} holds no {kind} weights
- **Observed state shown:** layout: {layout}
- **Fix:** use a checkpoint that holds its weights: *.safetensors files (Mistral's consolidated*.safetensors beside params.json), or one .gguf file beside a Transformers config and tokenizer
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_directory_without_weights_is_refused`

<a id="ster-model-checkpoint-hub-client"></a>

### `ster.model.checkpoint.hub-client`

- **What failed:** starting the Hugging Face Hub client for {model}
- **Cause:** the client could not be set up: {error}
- **Observed state shown:** HF_HOME: {hf_home}
- **Fix:** let this account create and write the Hugging Face cache directory, or point HF_HOME at one it can
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/hub.sh::a_hub_client_that_cannot_start_is_refused`

<a id="ster-model-checkpoint-repository-unknown"></a>

### `ster.model.checkpoint.repository-unknown`

- **What failed:** reading model repository {model} on the Hugging Face Hub
- **Cause:** the Hub answered {status}: no repository {model} at revision {revision} is visible to this account
- **Observed state shown:** the Hub's answer: {error}
- **Fix:** name a repository and revision that exist on https://huggingface.co, or a local checkpoint directory; for a gated or private repository, store a Hugging Face token in $HF_HOME/token (default ~/.cache/huggingface/token)
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/hub.sh::an_unknown_repository_is_refused`

<a id="ster-model-checkpoint-hub-unreachable"></a>

### `ster.model.checkpoint.hub-unreachable`

- **What failed:** reading model repository {model} on the Hugging Face Hub
- **Cause:** the Hub could not be reached, or answered with an error of its own: {error}
- **Observed state shown:** the Hub's answer: {error}
- **Fix:** run the command again once https://huggingface.co answers; the files already fetched stay in the Hugging Face cache
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/models/loading/hub.sh::an_unreachable_hub_is_refused`

<a id="ster-model-checkpoint-download-failed"></a>

### `ster.model.checkpoint.download-failed`

- **What failed:** downloading {file} of {model} from the Hugging Face Hub
- **Cause:** the file the repository lists could not be fetched: {error}
- **Observed state shown:** the Hub's answer: {error}
- **Fix:** run the command again; the files already fetched stay in the Hugging Face cache and are not fetched twice
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/models/loading/hub.sh::a_download_that_breaks_is_refused`

## `ster.model.weights`

map a checkpoint's safetensors or GGUF weights read-only at the run's dtype and build its decoder from them

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.model.weights`: 4 declared cause(s) of its own.

<a id="ster-model-weights-gguf-several"></a>

### `ster.model.weights.gguf-several`

- **What failed:** mapping the GGUF weights of the checkpoint
- **Cause:** a GGUF checkpoint is one .gguf file; this one lists {count}
- **Observed state shown:** files: {files}
- **Fix:** keep the one .gguf file the checkpoint is beside its config and tokenizer, and move the others out
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::two_gguf_files_are_refused`

<a id="ster-model-weights-gguf-unreadable"></a>

### `ster.model.weights.gguf-unreadable`

- **What failed:** reading GGUF weights {path}
- **Cause:** {path} is not a GGUF file Ster can read: {error}
- **Observed state shown:** path: {path}
- **Fix:** replace it with the complete .gguf file its publisher wrote; a partial download reads as this
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_file_that_is_not_gguf_is_refused`

<a id="ster-model-weights-safetensors-unmappable"></a>

### `ster.model.weights.safetensors-unmappable`

- **What failed:** mapping {count} safetensors weight file(s)
- **Cause:** a file is not safetensors Ster can map: {error}
- **Observed state shown:** files: {files}
- **Fix:** replace the file the sentence names with the complete one its publisher wrote; a partial download reads as this
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_weight_file_that_is_not_safetensors_is_refused`

<a id="ster-model-weights-weights-unfit"></a>

### `ster.model.weights.weights-unfit`

- **What failed:** building the decoder of {model} from its weights
- **Cause:** the weights do not fit the decoder its config describes: {error}
- **Observed state shown:** {shapes}
- **Fix:** use the config the weights were written with: a tensor's shape follows from hidden_size, intermediate_size, the head counts and vocab_size
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::weights_that_do_not_fit_the_config_are_refused`
