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

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.model.checkpoint`: 12 declared cause(s) of its own.

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
- **Cause:** the system broke off listing the directory although this account may read it: it was removed or replaced while Ster listed it, or the disk failed
- **Observed state shown:** path: {path} ({state}); error kind {kind}; the system answered: {error}
- **Fix:** see what is at {path} with `ls -ln {path}`, and run the command again once it lists
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/models/loading/load.sh::a_directory_listing_the_system_breaks_off_is_refused`

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
- **Cause:** the client's TLS connector could not be built: the system's TLS library or certificate store did not load
- **Observed state shown:** HF_HOME: {hf_home}; the client answered: {error}
- **Fix:** repair the system's TLS certificate store, which the client reads, or name a local checkpoint directory instead of a Hub id
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/hub.sh::a_hub_client_that_cannot_start_is_refused`

<a id="ster-model-checkpoint-repository-unknown"></a>

### `ster.model.checkpoint.repository-unknown`

- **What failed:** reading model repository {model} on the Hugging Face Hub
- **Cause:** the Hub answered {status}: no repository {model} at revision {revision} is visible to this account
- **Observed state shown:** GET {url} answered HTTP {status}: {error}
- **Fix:** name a repository and revision that exist on https://huggingface.co, or a local checkpoint directory; for a gated or private repository, store a Hugging Face token in $HF_HOME/token (default ~/.cache/huggingface/token)
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/hub.sh::an_unknown_repository_is_refused`

<a id="ster-model-checkpoint-hub-unreachable"></a>

### `ster.model.checkpoint.hub-unreachable`

- **What failed:** fetching {what} from the Hugging Face Hub
- **Cause:** no answer came from the Hub: the connection was not made or broke (no network, a name the resolver does not know, a proxy or TLS failing)
- **Observed state shown:** GET {url}: status: {status}; the client answered: {error}
- **Fix:** run the command again once https://huggingface.co answers from this machine; the files already fetched stay in the Hugging Face cache
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/models/loading/hub.sh::an_unreachable_hub_is_refused`

<a id="ster-model-checkpoint-hub-limited"></a>

### `ster.model.checkpoint.hub-limited`

- **What failed:** fetching {what} from the Hugging Face Hub
- **Cause:** the Hub answered 429: it limits how often this account or address may ask, and this request passed the limit
- **Observed state shown:** GET {url} answered HTTP {status}: {error}
- **Fix:** run the command again after a while, or store a Hugging Face token in $HF_HOME/token, which the Hub allows more requests
- **Code:** `rate_limit`, exit 69; retrying unchanged can help
- **Test:** `tests/models/loading/hub.sh::a_hub_that_limits_requests_is_refused`

<a id="ster-model-checkpoint-hub-failing"></a>

### `ster.model.checkpoint.hub-failing`

- **What failed:** fetching {what} from the Hugging Face Hub
- **Cause:** the Hub answered {status}: it failed on its side, so the request was right and the Hub could not serve it
- **Observed state shown:** GET {url} answered HTTP {status}: {error}
- **Fix:** run the command again once https://status.huggingface.co shows the Hub serving; the files already fetched stay in the Hugging Face cache
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/models/loading/hub.sh::a_hub_that_fails_on_its_side_is_refused`

<a id="ster-model-checkpoint-hub-refused"></a>

### `ster.model.checkpoint.hub-refused`

- **What failed:** fetching {what} from the Hugging Face Hub
- **Cause:** the Hub answered {status}, which says this request itself is wrong: a revision of another form, a file the Hub serves no more, or an address the client built from a model id the Hub does not read
- **Observed state shown:** model: {model}; revision: {revision}; GET {url} answered HTTP {status}: {error}
- **Fix:** check the model id and --revision against https://huggingface.co/{model}, then run again
- **Code:** `refused`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/hub.sh::a_request_the_hub_refuses_is_refused`

<a id="ster-model-checkpoint-download-uncached"></a>

### `ster.model.checkpoint.download-uncached`

- **What failed:** fetching {what} from the Hugging Face Hub
- **Cause:** the file arrived and could not be written into the Hugging Face cache {cache}: the cache directory is not writable for this account, or its disk is full
- **Observed state shown:** GET {url}; cache: {cache} ({cache_state}); the client answered: {error}
- **Fix:** let this account write {cache} and free space on its disk (`df -h {cache}`), or point HF_HOME at a writable directory
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/hub.sh::a_cache_the_client_cannot_write_is_refused`

## `ster.model.decoder`

build a family's own blocks from the weights and run them: the expert routing tables and scalars a checkpoint stores beside its projections, and what one layer leaves in a forward pass for a later one

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.model.decoder`: 13 declared cause(s) of its own.

<a id="ster-model-decoder-route-outside"></a>

### `ster.model.decoder.route-outside`

- **What failed:** reading the hash routing table of layer {layer}
- **Cause:** layer {layer}'s mlp.gate.tid2eid sends token {token} to expert {expert}, and the layer holds experts 0 to {count} exclusive
- **Observed state shown:** tid2eid row {token}: expert {expert}; experts: {count}
- **Fix:** fetch the checkpoint's weights again from where it was published; a routing table names only the experts its layer stores
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_routing_table_naming_an_expert_past_the_layer_is_refused`

<a id="ster-model-decoder-keys-unshared"></a>

### `ster.model.decoder.keys-unshared`

- **What failed:** attending at layer {layer} over the keys and values layer {source} leaves for it
- **Cause:** layer {layer} reuses the keys and values layer {source} stores earlier in the same forward pass (Gemma 4's key-value sharing, IQuest-LoopCoder's later loops), and layer {source} stored none in this pass: the decoder ran layer {layer} without running layer {source} first
- **Observed state shown:** layer {layer} reads layer {source}'s keys; layer {source}'s slot: empty
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder runs every layer in order, so this is a defect of Ster, not of the checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_layer_reusing_keys_its_source_did_not_store_is_refused`

<a id="ster-model-decoder-shortcut-unopened"></a>

### `ster.model.decoder.shortcut-unopened`

- **What failed:** adding LongCat-Flash's shortcut experts at layer {layer}
- **Cause:** the second half-layer {layer} adds the shortcut experts' output its first half, layer {first}, leaves in the same forward pass, and layer {first} left none: the decoder ran layer {layer} without running layer {first} first
- **Observed state shown:** layer {layer} closes the shortcut; the output layer {first} keeps: absent
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder runs every layer in order, so this is a defect of Ster, not of the checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_shortcut_no_earlier_layer_opened_is_refused`

<a id="ster-model-decoder-layer-form-unbuilt"></a>

### `ster.model.decoder.layer-form-unbuilt`

- **What failed:** running layer {layer} over {form}
- **Cause:** the decoder ran layer {layer} over {form}, and the layer was built without the weights that run them: the decoder and its layers were built from different readings of the config
- **Observed state shown:** layer {layer}; residual form: {form}; the layer's parts for it: absent
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder and every layer are built from one config, so this is a defect of Ster, not of the checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_layer_run_in_a_form_it_was_not_built_for_is_refused`

<a id="ster-model-decoder-ngram-rows"></a>

### `ster.model.decoder.ngram-rows`

- **What failed:** hashing LongCat-Flash-Lite's n-grams of row {row}
- **Cause:** this forward pass continues a decode whose cache kept the n-gram context of {kept} rows, and it runs row {row}: a cache continues only the batch that filled it
- **Observed state shown:** rows the cache kept: {kept}; row asked: {row}
- **Fix:** start a new cache (Runtime::cache) for a batch of another size; `ster` itself decodes one sequence per cache, so through the CLI report it at https://github.com/wisent-ai/ster/issues with the model and the command
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_cache_continued_by_a_wider_batch_is_refused`

<a id="ster-model-decoder-token-unrouted"></a>

### `ster.model.decoder.token-unrouted`

- **What failed:** routing token id {id} through a hash layer's mlp.gate.tid2eid
- **Cause:** the tokenizer spelled token id {id}, and the checkpoint's routing table names experts for token ids 0 to {rows} exclusive: the tokenizer is not the one the table was written for
- **Observed state shown:** token id: {id}; rows of tid2eid: {rows}
- **Fix:** load the checkpoint with the tokenizer it was published with; a hash-routed layer routes only the ids its tid2eid has a row for
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_token_past_the_routing_table_is_refused`

<a id="ster-model-decoder-token-ids-unset"></a>

### `ster.model.decoder.token-ids-unset`

- **What failed:** routing {tokens} tokens by token id
- **Cause:** a hash-routed layer chooses its experts by the ids of the tokens it runs, and the forward pass gave it {given} ids for {tokens} tokens: the decoder records every call's ids before its first layer
- **Observed state shown:** token ids given: {given}; tokens: {tokens}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: the decoder records the ids of every call that reaches a hash layer, so this is a defect of Ster, not of the checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_hash_layer_without_token_ids_is_refused`

<a id="ster-model-decoder-xielu-unheld"></a>

### `ster.model.decoder.xielu-unheld`

- **What failed:** applying xIELU outside a dense feed-forward
- **Cause:** xIELU takes its parameters from the dense feed-forward that holds them (mlp.act_fn), and a part that holds none applied it: the config reading accepts xielu only on Apertus, whose dense feed-forward applies it itself
- **Observed state shown:** activation: xielu, applied to a tensor of shape {shape}; parameters: none held
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::xielu_outside_a_dense_feed_forward_is_refused`

<a id="ster-model-decoder-entries-unmatched"></a>

### `ster.model.decoder.entries-unmatched`

- **What failed:** attending over layer {layer}'s compressed entries
- **Cause:** DeepSeek-V4's indexer ranked {ranked} compressed entries of layer {layer} and the compressor holds {held}: both pool the same positions at the same ratio, so the cache they continue from was filled by another sequence
- **Observed state shown:** entries ranked: {ranked}; entries held: {held}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::compressed_entries_the_indexer_did_not_rank_are_refused`

<a id="ster-model-decoder-logits-unread"></a>

### `ster.model.decoder.logits-unread`

- **What failed:** reading the logits of a forward pass whose residual stream is {hidden}
- **Cause:** the pass was run with the readout {readout}, which projects nothing onto the vocabulary, and its caller reads the vocabulary's logits: every reader of logits runs a readout of the last position or of every position
- **Observed state shown:** readout: {readout}; logits: none; residual stream shape {hidden}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::logits_read_from_a_hidden_readout_are_refused`

<a id="ster-model-decoder-activations-unrequested"></a>

### `ster.model.decoder.activations-unrequested`

- **What failed:** reading the activations of one forward pass
- **Cause:** the pass returned activations at another set of layers than it was asked for, or an empty one: the decoder keeps exactly the requested layers' last-position states, so a run would fit or export a direction on fewer reads than prompts
- **Observed state shown:** requested layers: {requested}; returned layers: {returned}; empty at: {empty}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the checkpoint or the prompts
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::activations_of_other_layers_are_refused`

<a id="ster-model-decoder-rows-empty"></a>

### `ster.model.decoder.rows-empty`

- **What failed:** {pass}
- **Cause:** the pass was handed no token rows, or a row of no tokens: every row is an encode's output, and encoding refuses a text the tokenizer turns into no tokens
- **Observed state shown:** row lengths: {lengths}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the prompts
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::a_pass_over_no_tokens_is_refused`

<a id="ster-model-decoder-positions-unkept"></a>

### `ster.model.decoder.positions-unkept`

- **What failed:** reading every token's state at layer {layer}
- **Cause:** the pass asked to keep each position's state at the layer kept another count of them than the text has tokens: the decoder keeps one state per token at every layer it is asked for
- **Observed state shown:** layer {layer}: {kept} states for {tokens} tokens
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the model and the command: this is a defect of Ster, not of the text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/families/decoder.sh::states_of_another_count_are_refused`

## `ster.model.t5`

run a T5 encoder-decoder network: the encoder reads the prompt, the decoder writes the answer, and steering, capture and adapters address the decoder's blocks

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.model.t5`: 23 declared cause(s) of its own.

<a id="ster-model-t5-t5-cross-attention-misplaced"></a>

### `ster.model.t5.t5-cross-attention-misplaced`

- **What failed:** running the T5 network's {stack}
- **Cause:** an encoder block was asked for or handed cross-attention memory, and only a decoder block attends to the encoder
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::an_encoder_block_with_cross_attention_is_refused`

<a id="ster-model-t5-t5-encoder-memory-missing"></a>

### `ster.model.t5.t5-encoder-memory-missing`

- **What failed:** running the T5 network's {stack}
- **Cause:** the decoder was run without the encoder memory its cross-attention reads: no source tokens were given and the cache holds none
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_decoder_without_encoder_memory_is_refused`

<a id="ster-model-t5-t5-encoder-memory-held"></a>

### `ster.model.t5.t5-encoder-memory-held`

- **What failed:** running the T5 network's {stack}
- **Cause:** source tokens were given to a cache that already holds the encoder memory of an earlier source
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_second_source_for_one_cache_is_refused`

<a id="ster-model-t5-t5-cache-empty"></a>

### `ster.model.t5.t5-cache-empty`

- **What failed:** running the T5 network's {stack}
- **Cause:** the key-value cache holds no layer, so it has no position to continue from
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_cache_of_no_layers_is_refused`

<a id="ster-model-t5-t5-cache-uneven"></a>

### `ster.model.t5.t5-cache-uneven`

- **What failed:** running the T5 network's {stack}
- **Cause:** the key-value cache's layers hold different numbers of positions: an earlier pass ended half done and its cache was reused
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_cache_of_uneven_layers_is_refused`

<a id="ster-model-t5-t5-position-mismatch"></a>

### `ster.model.t5.t5-position-mismatch`

- **What failed:** running the T5 network's {stack}
- **Cause:** the pass was given a starting position other than the number of positions its cache holds
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_position_the_cache_does_not_hold_is_refused`

<a id="ster-model-t5-t5-cache-layers-mismatch"></a>

### `ster.model.t5.t5-cache-layers-mismatch`

- **What failed:** running the T5 network's {stack}
- **Cause:** the key-value cache holds another number of layer slots than the decoder has blocks
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_cache_for_other_blocks_is_refused`

<a id="ster-model-t5-t5-cache-unsupported"></a>

### `ster.model.t5.t5-cache-unsupported`

- **What failed:** running the T5 network's {stack}
- **Cause:** a key-value cache was handed to a padded batch or a differentiable pass, and batched training and scoring run without one
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_cache_for_a_batch_or_training_is_refused`

<a id="ster-model-t5-t5-last-position-padded"></a>

### `ster.model.t5.t5-last-position-padded`

- **What failed:** running the T5 network's {stack}
- **Cause:** a padded batch asked for one common last position, and its rows end at different positions
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_padded_batch_read_at_one_position_is_refused`

<a id="ster-model-t5-t5-capture-batched"></a>

### `ster.model.t5.t5-capture-batched`

- **What failed:** running the T5 network's {stack}
- **Cause:** states were asked to be captured from several rows or a padded row, and a capture reads one unpadded sequence
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_capture_of_a_batch_is_refused`

<a id="ster-model-t5-t5-capture-outside"></a>

### `ster.model.t5.t5-capture-outside`

- **What failed:** running the T5 network's {stack}
- **Cause:** a capture names a layer past the decoder's blocks
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_capture_past_the_blocks_is_refused`

<a id="ster-model-t5-t5-steering-width"></a>

### `ster.model.t5.t5-steering-width`

- **What failed:** running the T5 network's {stack}
- **Cause:** the steering plan is of another width than the decoder's hidden states it is added to
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::steering_of_another_width_is_refused`

<a id="ster-model-t5-t5-lengths-without-tokens"></a>

### `ster.model.t5.t5-lengths-without-tokens`

- **What failed:** running the T5 network's {stack}
- **Cause:** encoder lengths were given without the encoder token rows they measure
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::lengths_without_tokens_are_refused`

<a id="ster-model-t5-t5-rows-mismatch"></a>

### `ster.model.t5.t5-rows-mismatch`

- **What failed:** running the T5 network's {stack}
- **Cause:** the encoder memory holds another number of rows than the decoder is run on, and both describe the same examples
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::encoder_and_decoder_rows_that_differ_are_refused`

<a id="ster-model-t5-t5-rows-empty"></a>

### `ster.model.t5.t5-rows-empty`

- **What failed:** running the T5 network's {stack}
- **Cause:** the {stack} was handed no token row or rows of no token, and a pass reads at least one token
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::rows_of_no_token_are_refused`

<a id="ster-model-t5-t5-lengths-count"></a>

### `ster.model.t5.t5-lengths-count`

- **What failed:** running the T5 network's {stack}
- **Cause:** the {stack} was handed another number of lengths than token rows, one length per row
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::lengths_for_other_rows_are_refused`

<a id="ster-model-t5-t5-padding-none"></a>

### `ster.model.t5.t5-padding-none`

- **What failed:** running the T5 network's {stack}
- **Cause:** a padding mask was asked for rows whose lengths list is empty
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_padding_mask_of_no_rows_is_refused`

<a id="ster-model-t5-t5-padding-outside"></a>

### `ster.model.t5.t5-padding-outside`

- **What failed:** running the T5 network's {stack}
- **Cause:** a row's valid length is zero or past the width of its tensor, so its attention would see no key or keys that are not there
- **Observed state shown:** {check}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the checkpoint: Ster builds these inputs itself, so this is a defect of Ster, not of the checkpoint or the prompt
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_padding_length_past_the_row_is_refused`

<a id="ster-model-t5-t5-size-overflow"></a>

### `ster.model.t5.t5-size-overflow`

- **What failed:** sizing {what} of the T5 network
- **Cause:** a position, shape or bucket count the pass computes is past this platform's sizes
- **Observed state shown:** {check}
- **Fix:** run on a shorter prompt; a sequence that long is past what any T5 checkpoint is trained at
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_size_past_the_platform_is_refused`

<a id="ster-model-t5-t5-memory-unreserved"></a>

### `ster.model.t5.t5-memory-unreserved`

- **What failed:** allocating {what} of the T5 network
- **Cause:** the host has no memory left for {what}
- **Observed state shown:** {cells} entries; the allocator answered: {error}
- **Fix:** run on a shorter prompt, or with fewer other programs holding the host's memory
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_table_past_the_host_memory_is_refused`

<a id="ster-model-t5-t5-sequence-past-context"></a>

### `ster.model.t5.t5-sequence-past-context`

- **What failed:** reading the T5 {stack}'s sequence
- **Cause:** the {stack} sequence reaches position {end}, past the {limit} positions the checkpoint's n_positions says it was trained at
- **Observed state shown:** {stack}: {end} positions; n_positions: {limit}
- **Fix:** shorten the prompt below {limit} tokens, or lower --max-new-tokens
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_prompt_past_n_positions_is_refused`

<a id="ster-model-t5-t5-context-unstated"></a>

### `ster.model.t5.t5-context-unstated`

- **What failed:** bounding the sequences of the T5 checkpoint
- **Cause:** the checkpoint's config.json states no n_positions, so Ster has no sequence length to bound its prompts and training rows by
- **Observed state shown:** n_positions: absent
- **Fix:** add n_positions, the length the checkpoint was trained at, to its config.json
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_checkpoint_without_n_positions_is_refused`

<a id="ster-model-t5-t5-pass-causal-only"></a>

### `ster.model.t5.t5-pass-causal-only`

- **What failed:** running {command} on the T5 checkpoint
- **Cause:** {command} scores one token sequence from its first position, and a T5 model reads the prompt in its encoder and writes the answer in its decoder, a split that pass does not carry
- **Observed state shown:** network: encoder-decoder (T5)
- **Fix:** run {command} on a causal-decoder checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/t5.sh::a_whole_sequence_pass_is_refused`

## `ster.model.weights`

map a checkpoint's safetensors or GGUF weights read-only at the run's dtype and build its decoder from them

Started through: run only by the functionalities that use it. Uses: `ster.model.decoder`, `ster.file`. Failure catalogue `ster.model.weights`: 8 declared cause(s) of its own.

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
- **Cause:** {path} is not a GGUF file: its header does not open with the GGUF magic, or names a version or value type Candle does not read
- **Observed state shown:** path: {path} ({state}); the reader answered: {error}
- **Fix:** replace it with the complete .gguf file its publisher wrote, or move it out of the checkpoint directory
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_file_that_is_not_gguf_is_refused`

<a id="ster-model-weights-gguf-truncated"></a>

### `ster.model.weights.gguf-truncated`

- **What failed:** reading GGUF weights {path}
- **Cause:** {path} ends before its header does: it was cut short, as a download that stopped leaves it
- **Observed state shown:** path: {path} ({state}); the reader answered: {error}
- **Fix:** fetch the complete .gguf file again from where it was published
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_gguf_file_cut_short_is_refused`

<a id="ster-model-weights-gguf-tensor-unreadable"></a>

### `ster.model.weights.gguf-tensor-unreadable`

- **What failed:** reading tensor {tensor} of GGUF weights {path}
- **Cause:** the file lists tensor {tensor} and its bytes cannot be read: the file ends before the tensor's offset, or stores it in a quantization Candle does not dequantize
- **Observed state shown:** path: {path} ({state}); asked for as {name}; the reader answered: {error}
- **Fix:** fetch the complete .gguf file again; a file whose quantization Candle does not read is published in another one beside it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::a_gguf_tensor_that_cannot_be_read_is_refused`

<a id="ster-model-weights-safetensors-unmappable"></a>

### `ster.model.weights.safetensors-unmappable`

- **What failed:** mapping {count} safetensors weight file(s)
- **Cause:** the system did not map the weight files into memory: the address space has no room for them, or the system refused the mapping
- **Observed state shown:** files: {files}; Candle answered: {error}
- **Fix:** close what holds the machine's memory, or load a smaller checkpoint; `vm_stat` (macOS) or `free -h` (Linux) shows what is left
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::weight_files_the_system_does_not_map_are_refused`

<a id="ster-model-weights-weights-unfit"></a>

### `ster.model.weights.weights-unfit`

- **What failed:** building the decoder of {model} from its weights
- **Cause:** a tensor the checkpoint stores has another shape than the decoder its config describes needs
- **Observed state shown:** {shapes}
- **Fix:** use the config the weights were written with: a tensor's shape follows from hidden_size, intermediate_size, the head counts and vocab_size
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::weights_that_do_not_fit_the_config_are_refused`

<a id="ster-model-weights-tensor-unstored"></a>

### `ster.model.weights.tensor-unstored`

- **What failed:** building the decoder of {model} from its weights
- **Cause:** the decoder its config describes needs a tensor the checkpoint does not store: the config names another architecture or layer count than the weights were written for, or a shard is missing
- **Observed state shown:** {shapes}
- **Fix:** use the config the weights were written with, and every shard model.safetensors.index.json lists
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::weights_without_a_tensor_the_config_needs_are_refused`

<a id="ster-model-weights-weights-unbuilt"></a>

### `ster.model.weights.weights-unbuilt`

- **What failed:** building the decoder of {model} from its weights
- **Cause:** the shapes fit, and the device did not build the decoder from them: it has no memory left for the weights, or does not compute in their dtype
- **Observed state shown:** {shapes}; Candle answered: {error}
- **Fix:** load at a smaller --precision, on `--device cpu`, or a smaller checkpoint
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/load.sh::weights_the_device_does_not_take_are_refused`
