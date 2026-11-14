# How loading a model with adapters fails

Every declared way loading a model fails once its checkpoint, config and
weights are read: placing fresh LoRA adapters on a family that cannot hold
them, and attaching a trained artifact made for another model, width or
purpose. Finding the checkpoint and mapping its weights are on
[model.md](model.md), reading its config on [model-config.md](model-config.md).
How a refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/model-load.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.model.tokenizer`

read a checkpoint's tokenizer.json, PLaMo tokenizer.jsonl or tiktoken vocabulary with the tokenizer_config.json beside it, and add its special tokens and BOS

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.model.tokenizer`: 23 declared cause(s) of its own.

<a id="ster-model-tokenizer-tokenizer-file-denied"></a>

### `ster.model.tokenizer.tokenizer-file-denied`

- **What failed:** reading {role} {path}
- **Cause:** the file cannot be read: this account may not read it
- **Observed state shown:** path: {path} ({state}); the system answered: {error}
- **Fix:** give this account read access to {path} (chmod u+r), or fetch the checkpoint's tokenizer files again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::an_unreadable_tokenizer_config_is_refused`

<a id="ster-model-tokenizer-tokenizer-file-unreadable"></a>

### `ster.model.tokenizer.tokenizer-file-unreadable`

- **What failed:** reading {role} {path}
- **Cause:** the file cannot be read although this account may read it: it was removed or replaced after the checkpoint directory was listed, or the disk failed while Ster read it
- **Observed state shown:** path: {path} ({state}); error kind {kind}; the system answered: {error}
- **Fix:** see what is at {path} with `ls -ln`, fetch the checkpoint's tokenizer files again, and run the command again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_tokenizer_file_removed_while_it_is_read_is_refused`

<a id="ster-model-tokenizer-vocabulary-not-text"></a>

### `ster.model.tokenizer.vocabulary-not-text`

- **What failed:** reading {role} {path}
- **Cause:** the file cannot be read as text: it holds bytes that are not UTF-8, and every vocabulary and tokenizer file Ster reads is UTF-8 text
- **Observed state shown:** path: {path} ({state}); the system answered: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_that_is_not_text_is_refused`

<a id="ster-model-tokenizer-tokenizer-json-malformed"></a>

### `ster.model.tokenizer.tokenizer-json-malformed`

- **What failed:** reading tokenizer {path}
- **Cause:** the tokenizers library cannot read it: it is not JSON, so it is no tokenizer.json a tokenizer was saved as
- **Observed state shown:** path: {path} ({state}); the JSON reader stopped at line {line}, column {column}: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_tokenizer_json_that_is_not_json_is_refused`

<a id="ster-model-tokenizer-tokenizer-json-unloadable"></a>

### `ster.model.tokenizer.tokenizer-json-unloadable`

- **What failed:** reading tokenizer {path}
- **Cause:** the tokenizers library cannot read it: it is JSON, and the model, normalizer, pre-tokenizer, post-processor or decoder it describes is missing or not one the library knows
- **Observed state shown:** path: {path}; top-level keys: {keys}; the library answered: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_tokenizer_json_that_does_not_load_is_refused`

<a id="ster-model-tokenizer-tokenizer-config-malformed"></a>

### `ster.model.tokenizer.tokenizer-config-malformed`

- **What failed:** reading tokenizer config {path}
- **Cause:** it is not JSON
- **Observed state shown:** path: {path} ({state}); the JSON reader stopped at line {line}, column {column}: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_tokenizer_config_that_is_not_json_is_refused`

<a id="ster-model-tokenizer-tokenizer-file-truncated"></a>

### `ster.model.tokenizer.tokenizer-file-truncated`

- **What failed:** reading {role} {path}
- **Cause:** it ends before its JSON value closes: the file was cut short, as a download that stopped or a disk that filled up leaves it
- **Observed state shown:** path: {path} ({state}); the JSON reader ran out at line {line}, column {column}: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_tokenizer_file_cut_short_is_refused`

<a id="ster-model-tokenizer-vocabulary-empty"></a>

### `ster.model.tokenizer.vocabulary-empty`

- **What failed:** reading vocabulary {path}
- **Cause:** it holds no {form} row, so the tokenizer would have no token to spell any text with
- **Observed state shown:** path: {path} ({state}); rows: 0
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::an_empty_vocabulary_is_refused`

<a id="ster-model-tokenizer-vocabulary-row-not-json"></a>

### `ster.model.tokenizer.vocabulary-row-not-json`

- **What failed:** reading vocabulary {path}
- **Cause:** line {line} is not JSON, and every line of a PLaMo tokenizer.jsonl is one JSON {form} row: the file was cut short or edited
- **Observed state shown:** line {line}: {text}; the JSON reader answered: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_row_that_is_not_json_is_refused`

<a id="ster-model-tokenizer-vocabulary-row-unscored"></a>

### `ster.model.tokenizer.vocabulary-row-unscored`

- **What failed:** reading vocabulary {path}
- **Cause:** line {line} is JSON and not a {form} row: it holds no piece string followed by a score number, which every PLaMo vocabulary row opens with
- **Observed state shown:** line {line}: {text}; first two entries: {first} and {second}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_row_without_its_score_is_refused`

<a id="ster-model-tokenizer-vocabulary-row-spaceless"></a>

### `ster.model.tokenizer.vocabulary-row-spaceless`

- **What failed:** reading vocabulary {path}
- **Cause:** line {line} holds no space, and a tiktoken vocabulary's row is a {form}: a base64 token, one space and its rank
- **Observed state shown:** line {line}: {text}; {characters} characters
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_row_without_a_space_is_refused`

<a id="ster-model-tokenizer-vocabulary-token-not-base64"></a>

### `ster.model.tokenizer.vocabulary-token-not-base64`

- **What failed:** reading vocabulary {path}
- **Cause:** the token {token} of line {line} is not base64, and a tiktoken vocabulary spells every token's bytes in base64 ({form} rows)
- **Observed state shown:** line {line}: {text}; the base64 reader answered: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_token_that_is_not_base64_is_refused`

<a id="ster-model-tokenizer-vocabulary-rank-not-id"></a>

### `ster.model.tokenizer.vocabulary-rank-not-id`

- **What failed:** reading vocabulary {path}
- **Cause:** the rank {rank} of line {line} is not a token id: a tiktoken vocabulary ranks each token by a whole number from 0 to {largest} ({form} rows)
- **Observed state shown:** line {line}: {text}; the number reader answered: {error}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_rank_that_is_not_an_id_is_refused`

<a id="ster-model-tokenizer-tokenizer-class-unnamed"></a>

### `ster.model.tokenizer.tokenizer-class-unnamed`

- **What failed:** choosing how to read vocabulary {path}
- **Cause:** its tokenizer config names no auto_map.AutoTokenizer or tokenizer_class string, so Ster cannot tell GLM-4's (ChatGLM4Tokenizer) from Kimi's (TikTokenTokenizer)
- **Observed state shown:** tokenizer config: {config}; auto_map.AutoTokenizer: {auto_map}; tokenizer_class: {class}
- **Fix:** keep the checkpoint's tokenizer_config.json beside it, or use a checkpoint that publishes tokenizer.json
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_tiktoken_vocabulary_without_its_class_is_refused`

<a id="ster-model-tokenizer-tokenizer-class-unsupported"></a>

### `ster.model.tokenizer.tokenizer-class-unsupported`

- **What failed:** choosing how to read vocabulary {path}
- **Cause:** it is a tiktoken or SentencePiece vocabulary for {class}, and Ster reads one only as GLM-4's (ChatGLM4Tokenizer) or Kimi's (TikTokenTokenizer)
- **Observed state shown:** tokenizer class: {class}; tokenizer config: {config}
- **Fix:** use a checkpoint that publishes tokenizer.json
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_of_another_class_is_refused`

<a id="ster-model-tokenizer-bos-unnamed"></a>

### `ster.model.tokenizer.bos-unnamed`

- **What failed:** starting every sequence of {path} with its BOS token
- **Cause:** the tokenizer config sets add_bos_token and names no bos_token string
- **Observed state shown:** bos_token: {bos}; add_bos_token: true
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_bos_token_the_config_does_not_name_is_refused`

<a id="ster-model-tokenizer-prefix-absent"></a>

### `ster.model.tokenizer.prefix-absent`

- **What failed:** starting every sequence of {path} with {token}
- **Cause:** the vocabulary holds no token {token}
- **Observed state shown:** vocabulary: {path}, {size} tokens and added tokens, none spelled {token}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_bos_token_the_vocabulary_lacks_is_refused`

<a id="ster-model-tokenizer-split-uncompilable"></a>

### `ster.model.tokenizer.split-uncompilable`

- **What failed:** isolating the stretches {key} names in {path}
- **Cause:** the tokenizer config's {key} {threshold} asks for the split {pattern}, which the regular expression engine the tokenizers library splits with does not compile: a repetition count that large is past its limit
- **Observed state shown:** {key}: {threshold}; the engine answered: {error}
- **Fix:** set {key} in the tokenizer_config.json beside it to the threshold its publisher wrote, or remove the key so no stretch is isolated
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_split_threshold_past_the_engine_is_refused`

<a id="ster-model-tokenizer-special-outside"></a>

### `ster.model.tokenizer.special-outside`

- **What failed:** adding the special tokens of {path}
- **Cause:** the tokenizer config numbers a special token {id}, outside the {count} ids after the vocabulary's {base} ranks; Ster adds them in order
- **Observed state shown:** added_tokens_decoder: {id}; the ids Ster adds: {base} to {end} exclusive
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_special_token_past_the_reserved_ids_is_refused`

<a id="ster-model-tokenizer-added-tokens-not-object"></a>

### `ster.model.tokenizer.added-tokens-not-object`

- **What failed:** adding the special tokens of {path}
- **Cause:** the tokenizer config's added_tokens_decoder is not an object of id to token, the form Transformers writes it in
- **Observed state shown:** added_tokens_decoder: {value}; config: {config}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::added_tokens_that_are_not_an_object_are_refused`

<a id="ster-model-tokenizer-added-token-key-not-id"></a>

### `ster.model.tokenizer.added-token-key-not-id`

- **What failed:** adding the special tokens of {path}
- **Cause:** the tokenizer config's added_tokens_decoder keys an entry {key}, which is not a token id: each key is the id, a whole number, of the token its entry spells
- **Observed state shown:** entry {key}: {value}; config: {config}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::an_added_token_keyed_by_no_id_is_refused`

<a id="ster-model-tokenizer-added-token-content-absent"></a>

### `ster.model.tokenizer.added-token-content-absent`

- **What failed:** adding the special tokens of {path}
- **Cause:** entry {key} of the tokenizer config's added_tokens_decoder has no `content` string, which spells the token the id stands for
- **Observed state shown:** entry {key}: {value}; config: {config}
- **Fix:** fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::an_added_token_without_its_content_is_refused`

<a id="ster-model-tokenizer-tokenizer-model-unbuilt"></a>

### `ster.model.tokenizer.tokenizer-model-unbuilt`

- **What failed:** building the {model} of vocabulary {path}
- **Cause:** the tokenizers library refused the {model} Ster built from the vocabulary's {rows} rows: Ster builds it only from tokens the vocabulary holds, so the library this build carries reads that model otherwise than Ster expects
- **Observed state shown:** path: {path}; rows: {rows}; {built}; the library answered: {error}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the checkpoint and `ster --version`: this is a defect of Ster, not of the vocabulary
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/tokenizer.sh::a_vocabulary_the_library_does_not_build_is_refused`

## `ster.model.load`

load a local or Hugging Face checkpoint into Ster's runtime, place fresh LoRA adapters on it or attach a trained LoRA or reward artifact when one is named, refusing adapters its family cannot hold and an artifact trained for another model, width or purpose

Started through: run only by the functionalities that use it. Uses: `ster.runtime.device`, `ster.model.checkpoint`, `ster.model.config`, `ster.model.tokenizer`, `ster.model.weights`, `ster.lora.artifact`, `ster.runtime.layers`, `ster.model.adapters`. Failure catalogue `ster.model.load`: 3 declared cause(s) of its own.

<a id="ster-model-load-artifact-kind"></a>

### `ster.model.load.artifact-kind`

- **What failed:** attaching artifact {path} to {model}
- **Cause:** adapter artifact is a {found}, not a {wanted}
- **Observed state shown:** kind: {kind}
- **Fix:** attach a {wanted} artifact here: `ster tune inspect {path}` prints an artifact's kind
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/adapter.sh::a_reward_model_given_as_a_generation_adapter_is_refused`

<a id="ster-model-load-artifact-model"></a>

### `ster.model.load.artifact-model`

- **What failed:** attaching artifact {path} to {model}
- **Cause:** adapter was trained for model {trained}, current model is {model}
- **Observed state shown:** trained for: {trained}
- **Fix:** load the model the artifact was trained for with --model {trained}, or train an artifact for {model}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/adapter.sh::an_adapter_for_another_model_is_refused`

<a id="ster-model-load-artifact-width"></a>

### `ster.model.load.artifact-width`

- **What failed:** attaching artifact {path} to {model}
- **Cause:** adapter width {trained} does not match model width {width}
- **Observed state shown:** hidden_size: {width}
- **Fix:** train an artifact for {model}; an artifact attaches only to the width it was trained at
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/adapter.sh::an_adapter_of_another_width_is_refused`

## `ster.model.adapters`

place LoRA adapters on the projections a model's family stores as one tensor on every layer, refusing a family that holds none and a target it does not store

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.model.adapters`: 15 declared cause(s) of its own.

<a id="ster-model-adapters-adapters-short-convolution"></a>

### `ster.model.adapters.adapters-short-convolution`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's blocks include short-convolution mixers (LFM2) with no attention projection to adapt on every layer; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_short_convolution_model_is_refused`

<a id="ster-model-adapters-adapters-layer-plans"></a>

### `ster.model.adapters.adapters-layer-plans`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's layers differ in key-value heads and feed-forward width, and some have neither (DeciLM's block_configs), so no projection has one shape on every layer; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_model_of_per_layer_plans_is_refused`

<a id="ster-model-adapters-adapters-shortcut-experts"></a>

### `ster.model.adapters.adapters-shortcut-experts`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's layers each hold two attention and feed-forward pairs beside shortcut-connected experts (LongCat-Flash), so no projection sits at one name per Ster layer; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_shortcut_experts_model_is_refused`

<a id="ster-model-adapters-adapters-layers-repeated"></a>

### `ster.model.adapters.adapters-layers-repeated`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model runs its layers more than once (Nanbeige's num_loops, IQuest-LoopCoder's loop_num, HRM-Text's H_cycles and L_cycles), so one adapter would correct every pass of its layer; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_model_running_its_layers_twice_is_refused`

<a id="ster-model-adapters-adapters-streams-hyper-connected"></a>

### `ster.model.adapters.adapters-streams-hyper-connected`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's residual is several hyper-connected streams (DeepSeek-V4's and GLM-5-Next's hc_mult), which no adapter on one sublayer's projection sees whole; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_hyper_connected_model_is_refused`

<a id="ster-model-adapters-adapters-inkling-projections"></a>

### `ster.model.adapters.adapters-inkling-projections`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's attention and feed-forward are Inkling's own (wq_du, wk_dv, wv_dv, wo_ud, w13_dn, w2_md behind short convolutions), which Ster's adapters do not attach to; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::an_inkling_model_is_refused`

<a id="ster-model-adapters-adapters-lightning-attention"></a>

### `ster.model.adapters.adapters-lightning-attention`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's blocks include lightning-attention mixers (MiniMax-Text-01's and Ling 2.5's linear attention) with no attention projection to adapt on every layer; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_lightning_attention_model_is_refused`

<a id="ster-model-adapters-adapters-delta-rule"></a>

### `ster.model.adapters.adapters-delta-rule`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's blocks include gated delta-rule mixers (Qwen3-Next's and Kimi-Linear's linear attention) with no attention projection to adapt on every layer; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_delta_rule_model_is_refused`

<a id="ster-model-adapters-adapters-state-space"></a>

### `ster.model.adapters.adapters-state-space`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model's blocks are state-space mixers (Mamba) with no attention or feed-forward projection to adapt; Ster steers it but trains no adapters on it
- **Observed state shown:** the family's {feature}; targets asked for: {targets}
- **Fix:** steer this model with a steering vector (`ster vector train --model {model}`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_model_that_trains_no_adapters_is_refused`

<a id="ster-model-adapters-target-query-bottlenecked"></a>

### `ster.model.adapters.target-query-bottlenecked`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model has no single {target} projection to adapt because its query passes a bottleneck and a norm (Step3's q_proj and inter_norm) before wq
- **Observed state shown:** targets this model has: {choices}; targets asked for: {targets}
- **Fix:** choose adapter targets among {choices} with --targets
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_bottlenecked_query_is_refused`

<a id="ster-model-adapters-target-heads-differ"></a>

### `ster.model.adapters.target-heads-differ`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model has no single {target} projection to adapt because its full-attention layers' heads differ from its sliding-window layers' in count or width (Gemma 4's global_head_dim, MiMo-V2's swa_num_key_value_heads, Step 3.5's attention_other_setting, Dots3-Note's swa_* latent attention)
- **Observed state shown:** targets this model has: {choices}; targets asked for: {targets}
- **Fix:** choose adapter targets among {choices} with --targets
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::an_attention_target_of_two_head_shapes_is_refused`

<a id="ster-model-adapters-target-value-experts"></a>

### `ster.model.adapters.target-value-experts`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model has no single {target} projection to adapt because its routed layers take their value from value experts (K2-Horizon's v_experts) with no v_proj
- **Observed state shown:** targets this model has: {choices}; targets asked for: {targets}
- **Fix:** choose adapter targets among {choices} with --targets
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_value_of_value_experts_is_refused`

<a id="ster-model-adapters-target-attention-latent"></a>

### `ster.model.adapters.target-attention-latent`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model has no single {target} projection to adapt because its attention is latent (DeepSeek's low-rank query and key-value), with no query, key or value projection of the head width
- **Observed state shown:** targets this model has: {choices}; targets asked for: {targets}
- **Fix:** choose adapter targets among {choices} with --targets
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_latent_attention_target_is_refused`

<a id="ster-model-adapters-target-feed-forward-experts"></a>

### `ster.model.adapters.target-feed-forward-experts`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model has no single {target} projection to adapt because its feed-forward is a mixture of experts, one gate, up and down projection per expert
- **Observed state shown:** targets this model has: {choices}; targets asked for: {targets}
- **Fix:** choose adapter targets among {choices} with --targets
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_feed_forward_target_of_experts_is_refused`

<a id="ster-model-adapters-target-feed-forward-ungated"></a>

### `ster.model.adapters.target-feed-forward-ungated`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model has no single {target} projection to adapt because its feed-forward has no gate
- **Observed state shown:** targets this model has: {choices}; targets asked for: {targets}
- **Fix:** choose adapter targets among {choices} with --targets
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_target_the_model_lacks_is_refused`
