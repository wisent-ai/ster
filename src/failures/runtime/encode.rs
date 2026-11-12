//! How turning text into this checkpoint's token ids fails
//! (src/runtime/text.rs): a prompt, a training example, a preference side,
//! a conversation and an answer label, each through the tokenizer the
//! checkpoint publishes and, when the run applies it, its chat template.

use wisent_errors::{Cause, Code, Operation};

use super::super::chat::TEMPLATE;
use super::DOCS;

pub static TEXT_BLANK: Cause = Cause {
    id: "text-blank",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "encoding the {what}",
    cause: "the {what} is empty or only whitespace, so there is nothing to encode",
    observed: "{what}: {text}",
    remedy: "give the {what} at least one word",
    test: "tests/runtime/encode.sh::a_blank_prompt_is_refused",
};

pub static TOKENS_NONE: Cause = Cause {
    id: "tokens-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "encoding the {what}",
    cause: "this model's tokenizer turns {text} into no tokens, so the model has nothing to read",
    observed: "{what}: {text}; token ids: []",
    remedy: "give the {what} text this model's tokenizer keeps, or use the checkpoint's own tokenizer.json if a normalizer added to it removes this text",
    test: "tests/runtime/encode.sh::text_the_tokenizer_drops_is_refused",
};

pub static TOKENIZER_FAILED: Cause = Cause {
    id: "tokenizer-failed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "encoding the {what}",
    cause: "the tokenizers library refused to encode the text: the tokenizer.json's normalizer, pre-tokenizer or model raised on it (a byte its vocabulary cannot represent, a pattern that fails on the text)",
    observed: "{what}: {text}; the library answered: {error}",
    remedy: "use the tokenizer.json the checkpoint was published with, unedited",
    test: "tests/runtime/encode.sh::a_text_the_tokenizer_raises_on_is_refused",
};

pub static PIECE_UNKNOWN: Cause = Cause {
    id: "piece-unknown",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "encoding the {what}",
    cause: "the text holds a piece the tokenizer's vocabulary lacks, and its tokenizer.json names no unknown token (unk_id) to stand in for it",
    observed: "{what}: {text}; the tokenizer holds {vocabulary} tokens; the library answered: {error}",
    remedy: "write the {what} with words the vocabulary holds, or use a tokenizer.json whose model names an unk_token",
    test: "tests/runtime/encode.sh::a_text_the_tokenizer_refuses_is_refused",
};

pub static SYSTEM_UNPLACEABLE: Cause = Cause {
    id: "system-unplaceable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "encoding a training example",
    cause: "the example has a system prompt, which only a chat template can place, and this run's chat template is {status}",
    observed: "system: {system}; chat template: {status}",
    remedy: "train a model whose tokenizer_config.json carries a chat_template with --chat-template auto, or remove the example's system key",
    test: "tests/tuning/runs/examples.sh::a_system_prompt_without_a_chat_template_is_refused",
};

pub static CONVERSATION_UNTEMPLATED: Cause = Cause {
    id: "conversation-untemplated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "encoding a conversation of several turns",
    cause: "only the model's chat template knows how turns are delimited, and this run's chat template is {status}",
    observed: "chat template: {status}",
    remedy: "converse with a model whose tokenizer_config.json carries a chat_template, with --chat-template auto",
    test: "tests/runtime/encode.sh::a_conversation_without_a_chat_template_is_refused",
};

pub static LABEL_UNSPELLED: Cause = Cause {
    id: "label-unspelled",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "finding the token of answer label {label}",
    cause: "this tokenizer has no single token for {label}, so the model's answer cannot be read from one position",
    observed: "spellings looked up: {spellings}",
    remedy: "use a model whose vocabulary holds each answer letter as one token; every Llama, Qwen and Mistral vocabulary does",
    test: "tests/runtime/encode.sh::a_label_the_vocabulary_lacks_is_refused",
};

/// Turning text into this checkpoint's token ids.
pub static ENCODE: Operation = Operation {
    id: "ster.runtime.encode",
    entries: &[],
    uses: &[&TEMPLATE],
    summary: "tokenize a prompt, a training example, a preference side, a conversation or an answer label the way this checkpoint expects, through its chat template when the run applies one",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &TEXT_BLANK,
        &TOKENS_NONE,
        &TOKENIZER_FAILED,
        &PIECE_UNKNOWN,
        &SYSTEM_UNPLACEABLE,
        &CONVERSATION_UNTEMPLATED,
        &LABEL_UNSPELLED,
    ],
};
