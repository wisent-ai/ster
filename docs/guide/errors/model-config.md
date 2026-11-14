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

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.model.config`: 13 declared cause(s) of its own.

<a id="ster-model-config-config-not-object"></a>

### `ster.model.config.config-not-object`

- **What failed:** reading model config {path}
- **Cause:** it is JSON, and not one object of settings: a config states each setting under its key
- **Observed state shown:** path: {path}; it holds {held}
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
- **Cause:** Candle's Llama config reader needs a key this config leaves out or states with another type than it reads, after every key Ster checks itself passed
- **Observed state shown:** path: {path}; the reader answered: {error}
- **Fix:** state the key the reader names with the type it names, as the publisher's config does
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_config_candle_cannot_read_is_refused`

<a id="ster-model-config-settings-unimplemented"></a>

### `ster.model.config.settings-unimplemented`

- **What failed:** reading the decoder's settings from the checkpoint's config
- **Cause:** the config states a variant of a setting (an activation, a norm, a router score, a rotation, a layer kind) that Ster's decoder for its family does not implement, so the decoder would compute another model than the weights were trained as
- **Observed state shown:** {detail}; the config states {observed}
- **Fix:** load a checkpoint of a variant Ster implements for the family (README.md lists the families Ster runs); `ster support report --description "…"` asks Ster's maintainers for this one
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_setting_ster_does_not_implement_is_refused`

<a id="ster-model-config-settings-contradictory"></a>

### `ster.model.config.settings-contradictory`

- **What failed:** reading the decoder's settings from the checkpoint's config
- **Cause:** two settings of the config disagree (a per-layer list of another length than the layers, heads a group count does not divide, widths that do not fit one another), so no decoder can be built that follows both
- **Observed state shown:** {detail}; the config states {observed}
- **Fix:** correct the key the observed sentence names to the value the checkpoint's publisher states: the weights' shapes show which of the two settings is right
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/settings.sh::an_indexer_narrower_than_the_rotation_is_refused`

<a id="ster-model-config-settings-out-of-range"></a>

### `ster.model.config.settings-out-of-range`

- **What failed:** reading the decoder's settings from the checkpoint's config
- **Cause:** a setting of the config holds a value outside the range the decoder runs (zero where at least one is needed, a width past the head, a layer index past the layers, a scale that is not positive)
- **Observed state shown:** {detail}; the config states {observed}
- **Fix:** state the setting within the range the observed sentence names, as the checkpoint's publisher does
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/settings.sh::a_rotation_wider_than_the_head_is_refused`

<a id="ster-model-config-settings-incomplete"></a>

### `ster.model.config.settings-incomplete`

- **What failed:** reading the decoder's settings from the checkpoint's config
- **Cause:** the config turns on a part of the decoder (latent attention, a state-space mixer, an expert router, a scaled rotation) and leaves out a setting that part is sized by, and the family gives that setting no default
- **Observed state shown:** {detail}; the config states {observed}
- **Fix:** state the settings the observed sentence names, as the checkpoint's publisher does; a config cut short or written by hand leaves them out
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/settings.sh::an_indexer_without_a_latent_query_is_refused`

<a id="ster-model-config-settings-malformed"></a>

### `ster.model.config.settings-malformed`

- **What failed:** reading the decoder's settings from the checkpoint's config
- **Cause:** a setting of the config is not in the form its family spells it (a per-layer mark other than the ones the family names, a list holding something other than numbers, an entry that is not a token id or an object)
- **Observed state shown:** {detail}; the config states {observed}
- **Fix:** state the setting in the form the observed sentence names, as the checkpoint's publisher does
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/settings.sh::a_layer_mark_out_of_form_is_refused`
