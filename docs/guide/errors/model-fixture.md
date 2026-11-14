# How creating a checkpoint from a fixture declaration fails

Every declared way `ster model fixture create` and
`ster request model/fixture/create` fail: reading the fixture declaration and
the Llama config and tokenizer it names, checking what they declare, and
writing the new checkpoint directory and verifying it by loading it and
forwarding the declared prompt. Reading the tokenizer declaration is on
[tokenizer.md](tokenizer.md), loading the written weights on
[model-load.md](model-load.md), and a file failing to be read or written on
[file.md](file.md). How a refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/model-fixture.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.model.fixture`

read a fixture declaration, its Llama config and tokenizer, check what they declare, write config.json, tokenizer.json and model.safetensors to a new directory, and verify the checkpoint by loading it and forwarding the declared prompt

Started through: `ster model fixture create`, `ster request model/fixture/create`. Uses: `ster.tokenizer.declaration`, `ster.model.load`, `ster.file`. Failure catalogue `ster.model.fixture`: 28 declared cause(s) of its own.

<a id="ster-model-fixture-input-missing"></a>

### `ster.model.fixture.input-missing`

- **What failed:** reading fixture {role} {path}
- **Cause:** no file is at {path}
- **Observed state shown:** path: {path} ({state})
- **Fix:** give the path of an existing fixture {role}; the config and tokenizer paths resolve relative to the declaration
- **Code:** `not_found`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_input_that_does_not_exist_is_refused`

<a id="ster-model-fixture-input-unreadable"></a>

### `ster.model.fixture.input-unreadable`

- **What failed:** reading fixture {role} {path}
- **Cause:** the file cannot be read: this account may not read it, or it is no regular file
- **Observed state shown:** path: {path} ({state}); error kind {kind}; the system answered: {error}
- **Fix:** give this account read access to {path} (chmod u+r), or name a regular file
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_unreadable_input_is_refused`

<a id="ster-model-fixture-input-not-json"></a>

### `ster.model.fixture.input-not-json`

- **What failed:** reading fixture {role} {path}
- **Cause:** it is not JSON
- **Observed state shown:** the JSON reader stopped at line {line}, column {column}: {error}
- **Fix:** docs/examples/model/fixture.json, config.json and vocabulary.json show each field; run `ster model fixture create --from <declaration> --output <new directory>` again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_input_that_is_not_json_is_refused`

<a id="ster-model-fixture-input-shape"></a>

### `ster.model.fixture.input-shape`

- **What failed:** reading fixture {role} {path}
- **Cause:** it is JSON, and a field the fixture {role} requires is missing, one it does not take is present, or one holds another type
- **Observed state shown:** at line {line}, column {column}: {error}
- **Fix:** docs/examples/model/fixture.json, config.json and vocabulary.json show each field; run `ster model fixture create --from <declaration> --output <new directory>` again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_input_of_another_shape_is_refused`

<a id="ster-model-fixture-model-type-unsupported"></a>

### `ster.model.fixture.model-type-unsupported`

- **What failed:** checking fixture config {path}
- **Cause:** the fixture tensor layout requires model_type=llama, and the config declares {model_type}
- **Observed state shown:** model_type: {model_type}
- **Fix:** set model_type to "llama" in {path}; the fixture writes Llama checkpoints only
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_config_of_another_family_is_refused`

<a id="ster-model-fixture-mlp-bias-unsupported"></a>

### `ster.model.fixture.mlp-bias-unsupported`

- **What failed:** checking fixture config {path}
- **Cause:** the config declares mlp_bias=true, but Ster's Llama consumer does not apply feed-forward biases; no checkpoint was written
- **Observed state shown:** {path}: "mlp_bias": true
- **Fix:** set mlp_bias to false in {path}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_feed_forward_bias_is_refused`

<a id="ster-model-fixture-heads-ungrouped"></a>

### `ster.model.fixture.heads-ungrouped`

- **What failed:** checking fixture config {path}
- **Cause:** the {heads} attention heads do not split into whole groups over the {kv} key/value heads, which grouped-query attention requires
- **Observed state shown:** num_attention_heads: {heads}; num_key_value_heads: {kv}; remainder {remainder}
- **Fix:** set num_key_value_heads in {path} to a divisor of {heads}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::heads_that_do_not_group_are_refused`

<a id="ster-model-fixture-setting-not-finite"></a>

### `ster.model.fixture.setting-not-finite`

- **What failed:** checking fixture {role} {path}
- **Cause:** {key} must be a finite number, and it is {value}
- **Observed state shown:** {key}: {value}
- **Fix:** give {key} in {path} a finite number
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_setting_that_is_not_finite_is_refused`

<a id="ster-model-fixture-scale-negative"></a>

### `ster.model.fixture.scale-negative`

- **What failed:** drawing fixture {field} from its normal distribution
- **Cause:** its scale, a standard deviation, is negative
- **Observed state shown:** {field}: mean {mean}, scale {scale}; the sampler answered: {error}
- **Fix:** give {field}.scale in {path} zero or a positive number
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_negative_scale_is_refused`

<a id="ster-model-fixture-bias-missing"></a>

### `ster.model.fixture.bias-missing`

- **What failed:** checking fixture declaration {path}
- **Cause:** initialization.bias is required because the config declares attention_bias=true, and the declaration gives none
- **Observed state shown:** {config}: "attention_bias": true; {path}: initialization.bias absent
- **Fix:** give initialization.bias in {path} a mean and scale, or set attention_bias to false in the config
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_attention_bias_without_its_initialization_is_refused`

<a id="ster-model-fixture-prompt-blank"></a>

### `ster.model.fixture.prompt-blank`

- **What failed:** checking fixture declaration {path}
- **Cause:** verification.prompt is required: the checkpoint is verified by forwarding it, and it is blank
- **Observed state shown:** verification.prompt: {prompt}
- **Fix:** give verification.prompt in {path} a text the tokenizer spells
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_blank_verification_prompt_is_refused`

<a id="ster-model-fixture-layers-none"></a>

### `ster.model.fixture.layers-none`

- **What failed:** checking fixture declaration {path}
- **Cause:** verification.layers is required: the checkpoint is verified at those layers, and it lists none
- **Observed state shown:** verification.layers: []; the config declares {count} layers
- **Fix:** list in verification.layers of {path} at least one layer from 0 to {last}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::no_verification_layer_is_refused`

<a id="ster-model-fixture-layer-outside"></a>

### `ster.model.fixture.layer-outside`

- **What failed:** checking fixture declaration {path}
- **Cause:** verification layer {layer} lies outside {count} declared layers
- **Observed state shown:** verification.layers: {layers}; num_hidden_layers: {count}
- **Fix:** list in verification.layers of {path} only layers from 0 to {last}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_verification_layer_past_the_model_is_refused`

<a id="ster-model-fixture-tokenizer-unparsed"></a>

### `ster.model.fixture.tokenizer-unparsed`

- **What failed:** reading fixture tokenizer {path}
- **Cause:** the tokenizers library does not read it as a tokenizer.json: it is not JSON, or its model, normalizer or pre-tokenizer is missing or unknown to the library
- **Observed state shown:** path: {path}; the library answered: {error}
- **Fix:** name a tokenizer.json a tokenizer saved, or declare one with a tokenizer declaration (docs/guide/models/tokenizers.md)
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_tokenizer_the_library_does_not_read_is_refused`

<a id="ster-model-fixture-tokenizer-empty"></a>

### `ster.model.fixture.tokenizer-empty`

- **What failed:** reading fixture tokenizer {path}
- **Cause:** it holds no token, so no text could be spelled for the checkpoint
- **Observed state shown:** path: {path}; tokens: 0
- **Fix:** name a tokenizer that holds the tokens of the verification prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_empty_tokenizer_is_refused`

<a id="ster-model-fixture-token-outside"></a>

### `ster.model.fixture.token-outside`

- **What failed:** checking fixture tokenizer {path} against the config
- **Cause:** token {token} has id {id}, outside config vocab_size {size}, so the embedding would hold no row for it
- **Observed state shown:** token {token}: id {id}; vocab_size: {size}; highest id: {highest}
- **Fix:** set vocab_size in the config to at least {needed}, or give a tokenizer whose ids stay below {size}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_token_past_the_vocabulary_is_refused`

<a id="ster-model-fixture-tokenizer-too-wide"></a>

### `ster.model.fixture.tokenizer-too-wide`

- **What failed:** checking compiled fixture tokenizer {path} against the config
- **Cause:** the compiled tokenizer has {count} token identities, exceeding config vocab_size {size}
- **Observed state shown:** token identities: {count}; vocab_size: {size}
- **Fix:** set vocab_size in the config to at least {count}, or declare fewer tokens
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_declared_tokenizer_past_the_vocabulary_is_refused`

<a id="ster-model-fixture-width-overflow"></a>

### `ster.model.fixture.width-overflow`

- **What failed:** sizing the {what} projections of fixture config {path}
- **Cause:** head_dim {head_dim} times {heads} {what} heads overflows this host's {bits}-bit sizes
- **Observed state shown:** head_dim: {head_dim}; {what} heads: {heads}
- **Fix:** lower head_dim or the head count in {path}; a fixture is meant to be small
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_width_past_the_host_is_refused`

<a id="ster-model-fixture-output-not-utf8"></a>

### `ster.model.fixture.output-not-utf8`

- **What failed:** choosing fixture output {path}
- **Cause:** fixture output {path} cannot be passed to Ster's UTF-8 checkpoint loader: its name holds bytes that are not UTF-8
- **Observed state shown:** path: {path}
- **Fix:** give --output a directory whose path is UTF-8 text
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_output_named_in_bytes_that_are_not_text_is_refused`

<a id="ster-model-fixture-output-exists"></a>

### `ster.model.fixture.output-exists`

- **What failed:** choosing fixture output {path}
- **Cause:** fixture output {path} already exists; choose a new directory; existing files were not changed
- **Observed state shown:** path: {path} ({state})
- **Fix:** give --output a directory that does not exist yet
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_output_that_exists_is_refused`

<a id="ster-model-fixture-output-uninspectable"></a>

### `ster.model.fixture.output-uninspectable`

- **What failed:** choosing fixture output {path}
- **Cause:** Ster cannot tell whether {path} exists: this account may not look inside {parent}
- **Observed state shown:** {parent}: {parent_state}; error kind {kind}; the system answered: {error}
- **Fix:** give this account search access to {parent} (chmod u+x), or choose an output elsewhere
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_output_in_a_closed_directory_is_refused`

<a id="ster-model-fixture-output-denied"></a>

### `ster.model.fixture.output-denied`

- **What failed:** creating fixture {what} {path}
- **Cause:** this account may not create {path}: the directory above it does not let it write
- **Observed state shown:** {parent}: {parent_state}; the system answered: {error}; nothing was written
- **Fix:** give this account write access to {parent} (chmod u+w), or choose an output elsewhere
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_output_in_a_read_only_directory_is_refused`

<a id="ster-model-fixture-output-uncreatable"></a>

### `ster.model.fixture.output-uncreatable`

- **What failed:** creating fixture {what} {path}
- **Cause:** {path} cannot be created although this account may write beside it: a file stands where a directory of the path should be, or the disk is full or read-only
- **Observed state shown:** {parent}: {parent_state}; error kind {kind}; the system answered: {error}; nothing was written
- **Fix:** see what stands on the way with `ls -ln {parent}`, then choose an output whose parents are directories on a writable disk
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::an_output_below_a_file_is_refused`

<a id="ster-model-fixture-file-unwritten"></a>

### `ster.model.fixture.file-unwritten`

- **What failed:** writing fixture {what} {path}
- **Cause:** the disk refused the write after the output directory was created: it is full or read-only, or the directory was removed meanwhile
- **Observed state shown:** {bytes} bytes to write; error kind {kind}; the system answered: {error}
- **Fix:** free space on the disk holding {path}, or choose an output on another disk, and run the command again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_write_the_disk_refuses_is_refused`

<a id="ster-model-fixture-sample-not-finite"></a>

### `ster.model.fixture.sample-not-finite`

- **What failed:** building fixture tensor {name} of shape {shape}
- **Cause:** the normal distribution it is drawn from gave a value past what f32 holds: its scale is too large
- **Observed state shown:** {name}: element {index} drew {value}
- **Fix:** lower the scale of the initialization {name} is drawn from in the declaration
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_scale_past_f32_is_refused`

<a id="ster-model-fixture-tensor-unbuilt"></a>

### `ster.model.fixture.tensor-unbuilt`

- **What failed:** building fixture tensor {name} of shape {shape}
- **Cause:** this host cannot hold its {count} values: the declared widths ask for more memory than it has, or for more values than it can count
- **Observed state shown:** {name}: shape {shape}, {count} values; the allocator answered: {error}
- **Fix:** lower hidden_size, intermediate_size, vocab_size or the head counts in the config; a fixture is meant to be small
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_tensor_past_the_host_is_refused`

<a id="ster-model-fixture-verification-width"></a>

### `ster.model.fixture.verification-width`

- **What failed:** verifying fixture {path} at layer {layer}
- **Cause:** layer {layer} returned width {width}; Ster loaded hidden size {hidden}: the config Ster wrote and the loader that read it disagree on the width
- **Observed state shown:** layer {layer}: {width} values; hidden size loaded: {hidden}; the output was removed
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the declaration and the config: this is a defect of Ster, not of the declaration
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::a_layer_of_another_width_is_refused`

<a id="ster-model-fixture-verification-not-finite"></a>

### `ster.model.fixture.verification-not-finite`

- **What failed:** verifying fixture {path} at layer {layer}
- **Cause:** layer {layer} returned non-finite activation {value} at element {index}: the declared initialization scales grow the activations past what f32 holds
- **Observed state shown:** layer {layer}, element {index}: {value}; {count} of {width} values not finite; the output was removed
- **Fix:** lower the initialization scales in the declaration (initialization.projection, embedding) and create the fixture again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/fixture.sh::activations_that_overflow_are_refused`
