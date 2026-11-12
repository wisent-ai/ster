//! How one row of a vocabulary file, and one entry of the tokenizer config's
//! `added_tokens_decoder`, fail to be read (src/runtime/vocabulary.rs and
//! vocabulary/tiktoken.rs): one cause for each way a row or an entry is out
//! of its form.

use wisent_errors::{Cause, Code};

use super::RESTORE;

pub static ROW_NOT_JSON: Cause = Cause {
    id: "vocabulary-row-not-json",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading vocabulary {path}",
    cause: "line {line} is not JSON, and every line of a PLaMo tokenizer.jsonl is one JSON {form} row: the file was cut short or edited",
    observed: "line {line}: {text}; the JSON reader answered: {error}",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::a_vocabulary_row_that_is_not_json_is_refused",
};

pub static ROW_UNSCORED: Cause = Cause {
    id: "vocabulary-row-unscored",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading vocabulary {path}",
    cause: "line {line} is JSON and not a {form} row: it holds no piece string followed by a score number, which every PLaMo vocabulary row opens with",
    observed: "line {line}: {text}; first two entries: {first} and {second}",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::a_vocabulary_row_without_its_score_is_refused",
};

pub static ROW_SPACELESS: Cause = Cause {
    id: "vocabulary-row-spaceless",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading vocabulary {path}",
    cause: "line {line} holds no space, and a tiktoken vocabulary's row is a {form}: a base64 token, one space and its rank",
    observed: "line {line}: {text}; {characters} characters",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::a_vocabulary_row_without_a_space_is_refused",
};

pub static TOKEN_NOT_BASE64: Cause = Cause {
    id: "vocabulary-token-not-base64",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading vocabulary {path}",
    cause: "the token {token} of line {line} is not base64, and a tiktoken vocabulary spells every token's bytes in base64 ({form} rows)",
    observed: "line {line}: {text}; the base64 reader answered: {error}",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::a_vocabulary_token_that_is_not_base64_is_refused",
};

pub static RANK_NOT_ID: Cause = Cause {
    id: "vocabulary-rank-not-id",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading vocabulary {path}",
    cause: "the rank {rank} of line {line} is not a token id: a tiktoken vocabulary ranks each token by a whole number from 0 to {largest} ({form} rows)",
    observed: "line {line}: {text}; the number reader answered: {error}",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::a_vocabulary_rank_that_is_not_an_id_is_refused",
};

pub static ADDED_NOT_OBJECT: Cause = Cause {
    id: "added-tokens-not-object",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "adding the special tokens of {path}",
    cause: "the tokenizer config's added_tokens_decoder is not an object of id to token, the form Transformers writes it in",
    observed: "added_tokens_decoder: {value}; config: {config}",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::added_tokens_that_are_not_an_object_are_refused",
};

pub static ADDED_KEY_NOT_ID: Cause = Cause {
    id: "added-token-key-not-id",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "adding the special tokens of {path}",
    cause: "the tokenizer config's added_tokens_decoder keys an entry {key}, which is not a token id: each key is the id, a whole number, of the token its entry spells",
    observed: "entry {key}: {value}; config: {config}",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::an_added_token_keyed_by_no_id_is_refused",
};

pub static ADDED_CONTENT_ABSENT: Cause = Cause {
    id: "added-token-content-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "adding the special tokens of {path}",
    cause: "entry {key} of the tokenizer config's added_tokens_decoder has no `content` string, which spells the token the id stands for",
    observed: "entry {key}: {value}; config: {config}",
    remedy: RESTORE,
    test: "tests/models/tokenizer.sh::an_added_token_without_its_content_is_refused",
};
