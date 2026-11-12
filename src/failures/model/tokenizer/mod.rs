//! How reading a checkpoint's tokenizer fails (src/runtime/vocabulary.rs):
//! Transformers' tokenizer.json, PLaMo's tokenizer.jsonl, the tiktoken
//! vocabularies of GLM-4 and Kimi, and the tokenizer_config.json beside them.

use std::path::Path;

use wisent_errors::{Cause, Code, Operation};

use super::super::{file, refuse};

mod rows;

pub use rows::{
    ADDED_CONTENT_ABSENT, ADDED_KEY_NOT_ID, ADDED_NOT_OBJECT, RANK_NOT_ID, ROW_NOT_JSON,
    ROW_SPACELESS, ROW_UNSCORED, TOKEN_NOT_BASE64,
};

/// docs/guide/errors/model-load.md, beside loading with adapters.
const LOAD_DOCS: &str = page!("model-load");

const RESTORE: &str = "fetch the checkpoint's tokenizer files again from where it was published; Ster reads them as their publisher wrote them";

/// The refusal of reading `path`, the checkpoint's `role` (`tokenizer`,
/// `tokenizer config`, `vocabulary`), when the system answered `error`.
pub fn read_refusal(role: &str, path: &Path, error: &std::io::Error) -> anyhow::Error {
    let cause = match error.kind() {
        std::io::ErrorKind::PermissionDenied => &FILE_DENIED,
        std::io::ErrorKind::InvalidData => &VOCABULARY_NOT_TEXT,
        _ => &FILE_UNREADABLE,
    };
    let shown = path.display().to_string();
    refuse(
        cause,
        &[
            ("role", role),
            ("path", &shown),
            ("state", &file::state(path)),
            ("kind", &file::kind(error)),
            ("error", &error.to_string()),
        ],
    )
}

/// The refusal of `path`, the checkpoint's `role` (`tokenizer`, `tokenizer
/// config`), which the JSON reader stopped reading with `error`: `not_json`
/// for text that is not JSON, [`JSON_TRUNCATED`] for a value the file ends
/// inside.
pub fn json_refusal(
    not_json: &'static Cause,
    role: &str,
    path: &Path,
    error: &serde_json::Error,
) -> anyhow::Error {
    let cause = match error.classify() {
        serde_json::error::Category::Eof => &JSON_TRUNCATED,
        _ => not_json,
    };
    refuse(
        cause,
        &[
            ("role", role),
            ("path", &path.display().to_string()),
            ("state", &file::state(path)),
            ("line", &error.line().to_string()),
            ("column", &error.column().to_string()),
            ("error", &error.to_string()),
        ],
    )
}

pub static FILE_DENIED: Cause = Cause {
    id: "tokenizer-file-denied",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "the file cannot be read: this account may not read it",
    observed: "path: {path} ({state}); the system answered: {error}",
    remedy: "give this account read access to {path} (chmod u+r), or fetch the checkpoint's tokenizer files again",
    test: "tests/models/families/tokenizer.sh::an_unreadable_tokenizer_config_is_refused",
};

pub static FILE_UNREADABLE: Cause = Cause {
    id: "tokenizer-file-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "the file cannot be read although this account may read it: it was removed or replaced after the checkpoint directory was listed, or the disk failed while Ster read it",
    observed: "path: {path} ({state}); error kind {kind}; the system answered: {error}",
    remedy: "see what is at {path} with `ls -ln`, fetch the checkpoint's tokenizer files again, and run the command again",
    test: "tests/models/families/tokenizer.sh::a_tokenizer_file_removed_while_it_is_read_is_refused",
};

pub static VOCABULARY_NOT_TEXT: Cause = Cause {
    id: "vocabulary-not-text",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "the file cannot be read as text: it holds bytes that are not UTF-8, and every vocabulary and tokenizer file Ster reads is UTF-8 text",
    observed: "path: {path} ({state}); the system answered: {error}",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_vocabulary_that_is_not_text_is_refused",
};

pub static JSON_MALFORMED: Cause = Cause {
    id: "tokenizer-json-malformed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading tokenizer {path}",
    cause: "the tokenizers library cannot read it: it is not JSON, so it is no tokenizer.json a tokenizer was saved as",
    observed: "path: {path} ({state}); the JSON reader stopped at line {line}, column {column}: {error}",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_tokenizer_json_that_is_not_json_is_refused",
};

pub static JSON_UNLOADABLE: Cause = Cause {
    id: "tokenizer-json-unloadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading tokenizer {path}",
    cause: "the tokenizers library cannot read it: it is JSON, and the model, normalizer, pre-tokenizer, post-processor or decoder it describes is missing or not one the library knows",
    observed: "path: {path}; top-level keys: {keys}; the library answered: {error}",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_tokenizer_json_that_does_not_load_is_refused",
};

pub static CONFIG_MALFORMED: Cause = Cause {
    id: "tokenizer-config-malformed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading tokenizer config {path}",
    cause: "it is not JSON",
    observed: "path: {path} ({state}); the JSON reader stopped at line {line}, column {column}: {error}",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_tokenizer_config_that_is_not_json_is_refused",
};

pub static JSON_TRUNCATED: Cause = Cause {
    id: "tokenizer-file-truncated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "it ends before its JSON value closes: the file was cut short, as a download that stopped or a disk that filled up leaves it",
    observed: "path: {path} ({state}); the JSON reader ran out at line {line}, column {column}: {error}",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_tokenizer_file_cut_short_is_refused",
};

pub static VOCABULARY_EMPTY: Cause = Cause {
    id: "vocabulary-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading vocabulary {path}",
    cause: "it holds no {form} row, so the tokenizer would have no token to spell any text with",
    observed: "path: {path} ({state}); rows: 0",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::an_empty_vocabulary_is_refused",
};

pub static CLASS_UNNAMED: Cause = Cause {
    id: "tokenizer-class-unnamed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "choosing how to read vocabulary {path}",
    cause: "its tokenizer config names no auto_map.AutoTokenizer or tokenizer_class string, so Ster cannot tell GLM-4's (ChatGLM4Tokenizer) from Kimi's (TikTokenTokenizer)",
    observed: "tokenizer config: {config}; auto_map.AutoTokenizer: {auto_map}; tokenizer_class: {class}",
    remedy: "keep the checkpoint's tokenizer_config.json beside it, or use a checkpoint that publishes tokenizer.json",
    test: "tests/models/families/tokenizer.sh::a_tiktoken_vocabulary_without_its_class_is_refused",
};

pub static CLASS_UNSUPPORTED: Cause = Cause {
    id: "tokenizer-class-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "choosing how to read vocabulary {path}",
    cause: "it is a tiktoken or SentencePiece vocabulary for {class}, and Ster reads one only as GLM-4's (ChatGLM4Tokenizer) or Kimi's (TikTokenTokenizer)",
    observed: "tokenizer class: {class}; tokenizer config: {config}",
    remedy: "use a checkpoint that publishes tokenizer.json",
    test: "tests/models/families/tokenizer.sh::a_vocabulary_of_another_class_is_refused",
};

pub static BOS_UNNAMED: Cause = Cause {
    id: "bos-unnamed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting every sequence of {path} with its BOS token",
    cause: "the tokenizer config sets add_bos_token and names no bos_token string",
    observed: "bos_token: {bos}; add_bos_token: true",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_bos_token_the_config_does_not_name_is_refused",
};

pub static PREFIX_ABSENT: Cause = Cause {
    id: "prefix-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "starting every sequence of {path} with {token}",
    cause: "the vocabulary holds no token {token}",
    observed: "vocabulary: {path}, {size} tokens and added tokens, none spelled {token}",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_bos_token_the_vocabulary_lacks_is_refused",
};

pub static SPLIT_UNCOMPILABLE: Cause = Cause {
    id: "split-uncompilable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "isolating the stretches {key} names in {path}",
    cause: "the tokenizer config's {key} {threshold} asks for the split {pattern}, which the regular expression engine the tokenizers library splits with does not compile: a repetition count that large is past its limit",
    observed: "{key}: {threshold}; the engine answered: {error}",
    remedy: "set {key} in the tokenizer_config.json beside it to the threshold its publisher wrote, or remove the key so no stretch is isolated",
    test: "tests/models/families/tokenizer.sh::a_split_threshold_past_the_engine_is_refused",
};

pub static SPECIAL_OUTSIDE: Cause = Cause {
    id: "special-outside",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "adding the special tokens of {path}",
    cause: "the tokenizer config numbers a special token {id}, outside the {count} ids after the vocabulary's {base} ranks; Ster adds them in order",
    observed: "added_tokens_decoder: {id}; the ids Ster adds: {base} to {end} exclusive",
    remedy: RESTORE,
    test: "tests/models/families/tokenizer.sh::a_special_token_past_the_reserved_ids_is_refused",
};

pub static MODEL_UNBUILT: Cause = Cause {
    id: "tokenizer-model-unbuilt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "building the {model} of vocabulary {path}",
    cause: "the tokenizers library refused the {model} Ster built from the vocabulary's {rows} rows: Ster builds it only from tokens the vocabulary holds, so the library this build carries reads that model otherwise than Ster expects",
    observed: "path: {path}; rows: {rows}; {built}; the library answered: {error}",
    remedy: "report it at https://github.com/wisent-ai/ster/issues with the checkpoint and `ster --version`: this is a defect of Ster, not of the vocabulary",
    test: "tests/models/families/tokenizer.sh::a_vocabulary_the_library_does_not_build_is_refused",
};

/// Reading the tokenizer a checkpoint publishes.
pub static TOKENIZER: Operation = Operation {
    id: "ster.model.tokenizer",
    entries: &[],
    uses: &[],
    summary: "read a checkpoint's tokenizer.json, PLaMo tokenizer.jsonl or tiktoken vocabulary with the tokenizer_config.json beside it, and add its special tokens and BOS",
    docs: LOAD_DOCS,
    warnings: &[],
    causes: &[
        &FILE_DENIED,
        &FILE_UNREADABLE,
        &VOCABULARY_NOT_TEXT,
        &JSON_MALFORMED,
        &JSON_UNLOADABLE,
        &CONFIG_MALFORMED,
        &JSON_TRUNCATED,
        &VOCABULARY_EMPTY,
        &ROW_NOT_JSON,
        &ROW_UNSCORED,
        &ROW_SPACELESS,
        &TOKEN_NOT_BASE64,
        &RANK_NOT_ID,
        &CLASS_UNNAMED,
        &CLASS_UNSUPPORTED,
        &BOS_UNNAMED,
        &PREFIX_ABSENT,
        &SPLIT_UNCOMPILABLE,
        &SPECIAL_OUTSIDE,
        &ADDED_NOT_OBJECT,
        &ADDED_KEY_NOT_ID,
        &ADDED_CONTENT_ABSENT,
        &MODEL_UNBUILT,
    ],
};
