//! How compiling a native WordLevel tokenizer declaration fails
//! (src/tokenizer): reading the declaration, the identities it declares, the
//! tokenizer the library builds from it and the verification text run
//! through that tokenizer; and how `ster model tokenizer create` writes,
//! reads back and removes its output. A file that cannot be read or written,
//! or is not JSON of the declaration's form, is the `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::file::FILE;

/// docs/guide/errors/tokenizer.md, where these functionalities are listed.
const DOCS: &str = page!("tokenizer");

/// What a declaration is, for the `ster.file` refusals that name it.
pub const FORM: &str = "the tokenizer declaration of docs/guide/models/tokenizers.md: unknownToken, modelTokens, addedTokens, pipeline (normalizer, preTokenizer, postProcessor, decoder, each an object or null) and verification, no other field";

const CHECKING: &str = "checking the tokenizer declaration {path}";
const VERIFYING: &str = "running the verification text \"{text}\" through the tokenizer {source}";

pub static MISSING: Cause = Cause {
    id: "declaration-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading the tokenizer declaration {path}",
    cause: "no file is there: nothing at {path} is the declaration --from names",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name a tokenizer declaration, as docs/examples/model/vocabulary.json; `ls -ln {parent}` shows what that directory holds",
    test: "tests/models/declaration.sh::a_missing_declaration_is_refused",
};

pub static MODEL_TOKENS_NONE: Cause = Cause {
    id: "model-tokens-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer declaration {path} contains no modelTokens, so the model has no identity to read",
    observed: "modelTokens: []; addedTokens: {added}",
    remedy: "list the model's tokens in order under modelTokens, the unknown token among them",
    test: "tests/models/declaration.sh::a_declaration_without_model_tokens_is_refused",
};

pub static UNKNOWN_BLANK: Cause = Cause {
    id: "unknown-token-blank",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer declaration requires unknownToken: a WordLevel model reads every word it lacks as that token",
    observed: "unknownToken: \"\"; modelTokens: {count}",
    remedy: "set unknownToken to the content of one of modelTokens, such as \"[UNK]\"",
    test: "tests/models/declaration.sh::a_blank_unknown_token_is_refused",
};

pub static VERIFICATION_BLANK: Cause = Cause {
    id: "verification-blank",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer declaration requires verification.text: the saved tokenizer is proven by encoding and decoding it",
    observed: "verification.text: {bytes} bytes, all whitespace",
    remedy: "set verification.text to a sentence the vocabulary spells, such as one of its own tokens",
    test: "tests/models/declaration.sh::a_blank_verification_text_is_refused",
};

pub static UNKNOWN_UNDECLARED: Cause = Cause {
    id: "unknown-token-undeclared",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer unknownToken \"{unknown}\" is absent from its declared modelTokens, so the model has no id to read an unknown word as",
    observed: "unknownToken \"{unknown}\"; modelTokens: {count}, first {first}",
    remedy: "add \"{unknown}\" to modelTokens, or set unknownToken to one of them",
    test: "tests/models/declaration.sh::an_undeclared_unknown_token_is_refused",
};

pub static MODEL_TOKEN_BLANK: Cause = Cause {
    id: "model-token-blank",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer token at position {position} has empty content, and an empty token is no word to read",
    observed: "modelTokens[{position}].content: \"\"; the token before it: {before}",
    remedy: "give modelTokens[{position}] its content, or remove it (every later id moves down by one)",
    test: "tests/models/declaration.sh::a_blank_model_token_is_refused",
};

pub static MODEL_TOKEN_REPEATED: Cause = Cause {
    id: "model-token-repeated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer token \"{content}\" at position {position} duplicates earlier content; token identities must be unique, since one word has one id",
    observed: "modelTokens[{first}] and modelTokens[{position}] both hold \"{content}\"",
    remedy: "remove modelTokens[{position}] (every later id moves down by one), or give it another content",
    test: "tests/models/declaration.sh::a_repeated_model_token_is_refused",
};

pub static IDS_EXHAUSTED: Cause = Cause {
    id: "ids-exhausted",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer token at position {position} exceeds the native token id representation: ids are 32-bit, so a vocabulary holds at most {limit} tokens",
    observed: "modelTokens: {count}",
    remedy: "declare at most {limit} model tokens",
    test: "tests/models/declaration.sh::more_tokens_than_ids_are_refused",
};

pub static ADDED_TOKEN_BLANK: Cause = Cause {
    id: "added-token-blank",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer declaration {path} has empty addedTokens content at position {position}, and an empty token is no word to read",
    observed: "addedTokens[{position}].content: \"\"; addedTokens: {count}",
    remedy: "give addedTokens[{position}] its content, or remove it",
    test: "tests/models/declaration.sh::a_blank_added_token_is_refused",
};

pub static ADDED_TOKENS_MERGED: Cause = Cause {
    id: "added-tokens-merged",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: CHECKING,
    cause: "tokenizer declaration {path} requested {requested} added-token records, but the native operation applied {applied} and stored {stored} unique identities; addedTokens content must be unique",
    observed: "addedTokens content declared more than once: {repeated}",
    remedy: "declare each added token once; one content is one identity",
    test: "tests/models/declaration.sh::a_repeated_added_token_is_refused",
};

pub static UNSERIALIZED: Cause = Cause {
    id: "tokenizer-unserialized",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "writing the tokenizer compiled from {path} as JSON",
    cause: "the tokenizers library could not write the tokenizer it built as its JSON document: a pipeline stage holds data its own serializer does not take",
    observed: "pipeline: {pipeline}; the library answered: {error}",
    remedy: "declare each pipeline stage as the library's own JSON writes it, or null",
    test: "tests/models/declaration.sh::a_tokenizer_the_library_cannot_write_is_refused",
};

pub static UNLOADED: Cause = Cause {
    id: "tokenizer-unloaded",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "loading the tokenizer {source}",
    cause: "the tokenizers library cannot read the document it wrote: its writer and reader disagree on a pipeline stage",
    observed: "{bytes} bytes; the library answered: {error}",
    remedy: "declare the pipeline stage the answer names in another form, or null",
    test: "tests/models/declaration.sh::a_document_the_library_cannot_read_back_is_refused",
};

pub static UNENCODED: Cause = Cause {
    id: "verification-unencoded",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: VERIFYING,
    cause: "the tokenizer cannot encode the verification text: a pipeline stage raised on it, or the post-processor names a token the vocabulary lacks",
    observed: "addSpecialTokens: {special}; the library answered: {error}",
    remedy: "declare the pipeline stage the answer names so it takes the text, or set verification.addSpecialTokens to false",
    test: "tests/models/declaration.sh::a_text_the_tokenizer_cannot_encode_is_refused",
};

pub static IDENTITY_MISMATCH: Cause = Cause {
    id: "identity-mismatch",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: VERIFYING,
    cause: "the tokenizer encoded token \"{token}\" as id {id}, but its vocabulary maps that token to {declared}: a post-processor emits an id the vocabulary does not hold for it",
    observed: "token {position} of {tokens}: \"{token}\" = {id}; the vocabulary: {declared}",
    remedy: "give the post-processor's special tokens the ids the vocabulary holds for them, or add them to addedTokens",
    test: "tests/models/declaration.sh::a_post_processor_emitting_an_unmapped_id_is_refused",
};

pub static UNDECODED: Cause = Cause {
    id: "verification-undecoded",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: VERIFYING,
    cause: "the tokenizer cannot decode the ids it encoded the verification text as: its decoder raised on them",
    observed: "ids {ids}; skipSpecialTokens: {skip}; the library answered: {error}",
    remedy: "declare the decoder the answer names in another form, or null",
    test: "tests/models/declaration.sh::ids_the_decoder_raises_on_are_refused",
};

/// Compiling a tokenizer declaration into the library's tokenizer.
pub static DECLARATION: Operation = Operation {
    id: "ster.tokenizer.declaration",
    entries: &[],
    uses: &[&FILE],
    summary: "read a WordLevel tokenizer declaration, check its identities, build the native tokenizer and prove it by encoding and decoding the verification text",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &MISSING,
        &MODEL_TOKENS_NONE,
        &UNKNOWN_BLANK,
        &VERIFICATION_BLANK,
        &UNKNOWN_UNDECLARED,
        &MODEL_TOKEN_BLANK,
        &MODEL_TOKEN_REPEATED,
        &IDS_EXHAUSTED,
        &ADDED_TOKEN_BLANK,
        &ADDED_TOKENS_MERGED,
        &UNSERIALIZED,
        &UNLOADED,
        &UNENCODED,
        &IDENTITY_MISMATCH,
        &UNDECODED,
    ],
};

pub static OUTPUT_NOT_UTF8: Cause = Cause {
    id: "output-not-utf8",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the tokenizer output {path}",
    cause: "the output path is not UTF-8, and the answer names it as JSON text",
    observed: "--output {path} (shown with replacement characters)",
    remedy: "name the output by a UTF-8 path",
    test: "tests/models/declaration.sh::an_output_path_that_is_not_utf8_is_refused",
};

pub static OUTPUT_EXISTS: Cause = Cause {
    id: "output-exists",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "creating the tokenizer output {path}",
    cause: "something is already at {path}, and existing output is not replaced",
    observed: "path: {path} ({state}); nothing was written",
    remedy: "name a new --output, or remove {path} first if it is yours to replace",
    test: "tests/models/declaration.sh::an_existing_output_is_not_replaced",
};

pub static READBACK_DIFFERS: Cause = Cause {
    id: "readback-differs",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "reading back the tokenizer {path}",
    cause: "the bytes read back differ from the bytes written and synced: another process changed the file, or the disk returned other bytes",
    observed: "wrote {written} bytes, read {read} bytes; first difference at byte {offset}",
    remedy: "run the command again with a new --output; if it differs again, check the disk",
    test: "tests/models/declaration.sh::a_readback_that_differs_is_refused",
};

/// `ster model tokenizer create`: a declaration compiled and written.
pub static CREATE: Operation = Operation {
    id: "ster.tokenizer.create",
    entries: &["ster model tokenizer create", "ster request model/tokenizer/create"],
    uses: &[&DECLARATION, &FILE],
    summary: "compile the declaration, write it to a new file, read the file back and prove the saved tokenizer; a refused run removes only the file it created",
    docs: DOCS,
    warnings: &[],
    causes: &[&OUTPUT_NOT_UTF8, &OUTPUT_EXISTS, &READBACK_DIFFERS],
};

pub(super) static OPERATIONS: &[&Operation] = &[&DECLARATION, &CREATE];
