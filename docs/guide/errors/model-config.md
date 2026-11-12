# How reading a model's config fails

Every declared way reading a checkpoint's config into a decoder fails,
whichever command loads the model: the file, the `model_type` it names, the
keys every decoder needs, every key stated with the wrong type, and every
setting a family's decoder does not implement. Every key is read through one
reader: a key the config leaves out (or states as `null`) takes the default
the family's config class gives it, and a key stated with another type —
`"attention_bias": "yes"`, a `rope_scaling.rope_type` written as a number, a
`block_configs[3].ffn.ffn_mult` written as text — is refused as
`key-mistyped`, naming the key, the value and the type expected, never read
as if it were left out. A setting a family's reading cannot run is refused
through the cause of its reason: `settings-unimplemented` (a variant the
decoder does not implement), `settings-contradictory` (two settings that
disagree), `settings-out-of-range` (a value outside the range the decoder
runs), `settings-incomplete` (a part turned on without a setting it is sized
by) or `settings-malformed` (a setting not in the form the family spells it).
Each one's observed state is the sentence that names the config, the key and
what Ster implements, then what the config states for each key the sentence
names (`absent` for one it does not state). Finding the checkpoint
and mapping its weights are on [model.md](model.md). How a refusal reads and
which status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/model-config.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.model.config`

read a checkpoint's config (config.json, or Mistral's params.json put in its shape), choose the decoder family its model_type names, and read every setting that decoder needs, refusing one it does not implement

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.model.config`: 10 declared cause(s) of its own.

<a id="ster-model-config-config-unreadable"></a>

### `ster.model.config.config-unreadable`

- **What failed:** reading model config {path}
- **Cause:** the file cannot be read: {error}
- **Observed state shown:** path: {path}
- **Fix:** give this account read access to {path}, or name another checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::an_unreadable_config_is_refused`

<a id="ster-model-config-config-malformed"></a>

### `ster.model.config.config-malformed`

- **What failed:** reading model config {path}
- **Cause:** it is not a JSON object: {error}
- **Observed state shown:** path: {path}
- **Fix:** use the checkpoint's config as its publisher wrote it, one JSON object, as docs/examples/model/config.json is
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_config_that_is_not_json_is_refused`

<a id="ster-model-config-model-type-absent"></a>

### `ster.model.config.model-type-absent`

- **What failed:** reading the model type of {path}
- **Cause:** it states no model_type, and its architectures ({architectures}) name no model class Ster reads in its place
- **Observed state shown:** model_type: absent; architectures: {architectures}
- **Fix:** use a Hugging Face checkpoint whose model_type is one of {families}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_config_naming_no_model_type_is_refused`

<a id="ster-model-config-model-type-not-text"></a>

### `ster.model.config.model-type-not-text`

- **What failed:** reading the model type of {path}
- **Cause:** it states model_type as {value}, which is not a string
- **Observed state shown:** model_type: {value}
- **Fix:** use a Hugging Face checkpoint whose model_type is one of {families}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_model_type_that_is_not_text_is_refused`

<a id="ster-model-config-family-unknown"></a>

### `ster.model.config.family-unknown`

- **What failed:** choosing the decoder for {path}
- **Cause:** model architecture "{model_type}" is unsupported by this Ster build
- **Observed state shown:** model_type: "{model_type}"
- **Fix:** use a Hugging Face checkpoint whose model_type is one of {families}
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::an_unknown_model_type_is_refused`

<a id="ster-model-config-quantized"></a>

### `ster.model.config.quantized`

- **What failed:** choosing the decoder for {path}
- **Cause:** {path} is a quantized checkpoint (it declares quantization_config); Ster maps unquantized safetensors only
- **Observed state shown:** quantization_config: {value}
- **Fix:** use the checkpoint it was quantized from
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_quantized_checkpoint_is_refused`

<a id="ster-model-config-key-absent"></a>

### `ster.model.config.key-absent`

- **What failed:** reading the {model_type} decoder's {key} from {path}
- **Cause:** {path} states no {key}{spellings}; Ster needs the {model_type} decoder's {key} ({kind})
- **Observed state shown:** {key}: absent
- **Fix:** add {key} to the config as {kind}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_config_without_a_required_key_is_refused`

<a id="ster-model-config-key-mistyped"></a>

### `ster.model.config.key-mistyped`

- **What failed:** reading {key} from {path}
- **Cause:** {path} states {key} as {value}, which is not {kind}
- **Observed state shown:** {key}: {value}
- **Fix:** state {key} as {kind}, the type the publisher's config class reads it as
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_key_of_the_wrong_kind_is_refused`

<a id="ster-model-config-config-undecodable"></a>

### `ster.model.config.config-undecodable`

- **What failed:** reading the {model_type} decoder's settings from {path}
- **Cause:** Candle's Llama config does not read it: {error}
- **Observed state shown:** path: {path}
- **Fix:** state the key the reader names with the type it names, as the publisher's config does
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_config_candle_cannot_read_is_refused`

<a id="ster-model-config-settings-unrunnable"></a>

### `ster.model.config.settings-unrunnable`

- **What failed:** reading the decoder's settings from the checkpoint's config
- **Cause:** {detail}
- **Observed state shown:** the config states {observed}
- **Fix:** load a checkpoint whose config states settings Ster's decoder for its family implements, or, where the config contradicts itself, correct the key the sentence names; README.md lists the families Ster runs
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_setting_ster_does_not_implement_is_refused`
