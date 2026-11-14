# How compiling a tokenizer declaration fails

Every declared way compiling a native WordLevel tokenizer declaration fails
(the declaration of [Native tokenizer creation](../models/tokenizers.md)),
whether `ster model tokenizer create` writes it or `ster model fixture
create` composes it into a checkpoint, and how the create command writes,
reads back and removes its output. Reading a checkpoint's own tokenizer is
on [model.md](model.md); how a refusal reads and which status it exits with
is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/tokenizer.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.tokenizer.declaration`

read a WordLevel tokenizer declaration, check its identities, build the native tokenizer and prove it by encoding and decoding the verification text

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.tokenizer.declaration`: 15 declared cause(s) of its own.

<a id="ster-tokenizer-declaration-declaration-missing"></a>

### `ster.tokenizer.declaration.declaration-missing`

- **What failed:** reading the tokenizer declaration {path}
- **Cause:** no file is there: nothing at {path} is the declaration --from names
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name a tokenizer declaration, as docs/examples/model/vocabulary.json; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_missing_declaration_is_refused`

<a id="ster-tokenizer-declaration-model-tokens-none"></a>

### `ster.tokenizer.declaration.model-tokens-none`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer declaration {path} contains no modelTokens, so the model has no identity to read
- **Observed state shown:** modelTokens: []; addedTokens: {added}
- **Fix:** list the model's tokens in order under modelTokens, the unknown token among them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_declaration_without_model_tokens_is_refused`

<a id="ster-tokenizer-declaration-unknown-token-blank"></a>

### `ster.tokenizer.declaration.unknown-token-blank`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer declaration requires unknownToken: a WordLevel model reads every word it lacks as that token
- **Observed state shown:** unknownToken: ""; modelTokens: {count}
- **Fix:** set unknownToken to the content of one of modelTokens, such as "[UNK]"
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_blank_unknown_token_is_refused`

<a id="ster-tokenizer-declaration-verification-blank"></a>

### `ster.tokenizer.declaration.verification-blank`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer declaration requires verification.text: the saved tokenizer is proven by encoding and decoding it
- **Observed state shown:** verification.text: {bytes} bytes, all whitespace
- **Fix:** set verification.text to a sentence the vocabulary spells, such as one of its own tokens
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_blank_verification_text_is_refused`

<a id="ster-tokenizer-declaration-unknown-token-undeclared"></a>

### `ster.tokenizer.declaration.unknown-token-undeclared`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer unknownToken "{unknown}" is absent from its declared modelTokens, so the model has no id to read an unknown word as
- **Observed state shown:** unknownToken "{unknown}"; modelTokens: {count}, first {first}
- **Fix:** add "{unknown}" to modelTokens, or set unknownToken to one of them
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::an_undeclared_unknown_token_is_refused`

<a id="ster-tokenizer-declaration-model-token-blank"></a>

### `ster.tokenizer.declaration.model-token-blank`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer token at position {position} has empty content, and an empty token is no word to read
- **Observed state shown:** modelTokens[{position}].content: ""; the token before it: {before}
- **Fix:** give modelTokens[{position}] its content, or remove it (every later id moves down by one)
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_blank_model_token_is_refused`

<a id="ster-tokenizer-declaration-model-token-repeated"></a>

### `ster.tokenizer.declaration.model-token-repeated`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer token "{content}" at position {position} duplicates earlier content; token identities must be unique, since one word has one id
- **Observed state shown:** modelTokens[{first}] and modelTokens[{position}] both hold "{content}"
- **Fix:** remove modelTokens[{position}] (every later id moves down by one), or give it another content
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_repeated_model_token_is_refused`

<a id="ster-tokenizer-declaration-ids-exhausted"></a>

### `ster.tokenizer.declaration.ids-exhausted`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer token at position {position} exceeds the native token id representation: ids are 32-bit, so a vocabulary holds at most {limit} tokens
- **Observed state shown:** modelTokens: {count}
- **Fix:** declare at most {limit} model tokens
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::more_tokens_than_ids_are_refused`

<a id="ster-tokenizer-declaration-added-token-blank"></a>

### `ster.tokenizer.declaration.added-token-blank`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer declaration {path} has empty addedTokens content at position {position}, and an empty token is no word to read
- **Observed state shown:** addedTokens[{position}].content: ""; addedTokens: {count}
- **Fix:** give addedTokens[{position}] its content, or remove it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_blank_added_token_is_refused`

<a id="ster-tokenizer-declaration-added-tokens-merged"></a>

### `ster.tokenizer.declaration.added-tokens-merged`

- **What failed:** checking the tokenizer declaration {path}
- **Cause:** tokenizer declaration {path} requested {requested} added-token records, but the native operation applied {applied} and stored {stored} unique identities; addedTokens content must be unique
- **Observed state shown:** addedTokens content declared more than once: {repeated}
- **Fix:** declare each added token once; one content is one identity
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_repeated_added_token_is_refused`

<a id="ster-tokenizer-declaration-tokenizer-unserialized"></a>

### `ster.tokenizer.declaration.tokenizer-unserialized`

- **What failed:** writing the tokenizer compiled from {path} as JSON
- **Cause:** the tokenizers library could not write the tokenizer it built as its JSON document: a pipeline stage holds data its own serializer does not take
- **Observed state shown:** pipeline: {pipeline}; the library answered: {error}
- **Fix:** declare each pipeline stage as the library's own JSON writes it, or null
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_tokenizer_the_library_cannot_write_is_refused`

<a id="ster-tokenizer-declaration-tokenizer-unloaded"></a>

### `ster.tokenizer.declaration.tokenizer-unloaded`

- **What failed:** loading the tokenizer {source}
- **Cause:** the tokenizers library cannot read the document it wrote: its writer and reader disagree on a pipeline stage
- **Observed state shown:** {bytes} bytes; the library answered: {error}
- **Fix:** declare the pipeline stage the answer names in another form, or null
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_document_the_library_cannot_read_back_is_refused`

<a id="ster-tokenizer-declaration-verification-unencoded"></a>

### `ster.tokenizer.declaration.verification-unencoded`

- **What failed:** running the verification text "{text}" through the tokenizer {source}
- **Cause:** the tokenizer cannot encode the verification text: a pipeline stage raised on it, or the post-processor names a token the vocabulary lacks
- **Observed state shown:** addSpecialTokens: {special}; the library answered: {error}
- **Fix:** declare the pipeline stage the answer names so it takes the text, or set verification.addSpecialTokens to false
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_text_the_tokenizer_cannot_encode_is_refused`

<a id="ster-tokenizer-declaration-identity-mismatch"></a>

### `ster.tokenizer.declaration.identity-mismatch`

- **What failed:** running the verification text "{text}" through the tokenizer {source}
- **Cause:** the tokenizer encoded token "{token}" as id {id}, but its vocabulary maps that token to {declared}: a post-processor emits an id the vocabulary does not hold for it
- **Observed state shown:** token {position} of {tokens}: "{token}" = {id}; the vocabulary: {declared}
- **Fix:** give the post-processor's special tokens the ids the vocabulary holds for them, or add them to addedTokens
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::a_post_processor_emitting_an_unmapped_id_is_refused`

<a id="ster-tokenizer-declaration-verification-undecoded"></a>

### `ster.tokenizer.declaration.verification-undecoded`

- **What failed:** running the verification text "{text}" through the tokenizer {source}
- **Cause:** the tokenizer cannot decode the ids it encoded the verification text as: its decoder raised on them
- **Observed state shown:** ids {ids}; skipSpecialTokens: {skip}; the library answered: {error}
- **Fix:** declare the decoder the answer names in another form, or null
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::ids_the_decoder_raises_on_are_refused`

## `ster.tokenizer.create`

compile the declaration, write it to a new file, read the file back and prove the saved tokenizer; a refused run removes only the file it created

Started through: `ster model tokenizer create`, `ster request model/tokenizer/create`. Uses: `ster.tokenizer.declaration`, `ster.file`. Failure catalogue `ster.tokenizer.create`: 3 declared cause(s) of its own.

<a id="ster-tokenizer-create-output-not-utf8"></a>

### `ster.tokenizer.create.output-not-utf8`

- **What failed:** choosing the tokenizer output {path}
- **Cause:** the output path is not UTF-8, and the answer names it as JSON text
- **Observed state shown:** --output {path} (shown with replacement characters)
- **Fix:** name the output by a UTF-8 path
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::an_output_path_that_is_not_utf8_is_refused`

<a id="ster-tokenizer-create-output-exists"></a>

### `ster.tokenizer.create.output-exists`

- **What failed:** creating the tokenizer output {path}
- **Cause:** something is already at {path}, and existing output is not replaced
- **Observed state shown:** path: {path} ({state}); nothing was written
- **Fix:** name a new --output, or remove {path} first if it is yours to replace
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/declaration.sh::an_existing_output_is_not_replaced`

<a id="ster-tokenizer-create-readback-differs"></a>

### `ster.tokenizer.create.readback-differs`

- **What failed:** reading back the tokenizer {path}
- **Cause:** the bytes read back differ from the bytes written and synced: another process changed the file, or the disk returned other bytes
- **Observed state shown:** wrote {written} bytes, read {read} bytes; first difference at byte {offset}
- **Fix:** run the command again with a new --output; if it differs again, check the disk
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/models/declaration.sh::a_readback_that_differs_is_refused`
