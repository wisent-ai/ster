//! How a checkpoint's chat template fails (src/chat): choosing whether a run
//! applies it, reading it from tokenizer_config.json or chat_template.jinja,
//! compiling it, and rendering turns through it.

use wisent_errors::{Cause, Code, Operation};

use super::model::tokenizer::TOKENIZER;

/// docs/guide/errors/chat.md, where this functionality is listed.
const DOCS: &str = page!("chat");

const REFETCH: &str = "fetch the checkpoint's tokenizer_config.json and chat_template.jinja again, or run with --chat-template off to encode raw text";

pub static MODE_UNKNOWN: Cause = Cause {
    id: "mode-unknown",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the chat template mode",
    cause: "unknown chat template mode {value}; expected auto or off",
    observed: "--chat-template {value}",
    remedy: "give --chat-template auto or off",
    test: "tests/chat/template.sh::an_unknown_mode_is_refused",
};

pub static FORM_UNKNOWN: Cause = Cause {
    id: "template-form-unknown",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the chat_template of the tokenizer config",
    cause: "it is neither a template nor a list of them",
    observed: "chat_template: {value}; config: {path}",
    remedy: REFETCH,
    test: "tests/chat/template.sh::a_template_of_no_known_form_is_refused",
};

pub static FORM_UNCHOSEN: Cause = Cause {
    id: "template-default-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the chat_template of the tokenizer config",
    cause: "it lists several named templates and none of them is the default, and only the default is the conversation format",
    observed: "names listed: {names}; config: {path}",
    remedy: REFETCH,
    test: "tests/chat/template.sh::a_template_list_without_a_default_is_refused",
};

pub static FORM_EMPTY: Cause = Cause {
    id: "template-entry-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the chat_template of the tokenizer config",
    cause: "its chosen entry {name} holds no template string",
    observed: "entry: {entry}; config: {path}",
    remedy: REFETCH,
    test: "tests/chat/template.sh::a_template_entry_without_text_is_refused",
};

pub static TOKEN_UNDECLARED: Cause = Cause {
    id: "token-undeclared",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "compiling the chat template",
    cause: "the template writes {name}, but the tokenizer config declares none, so every rendering would lack it",
    observed: "{name}: absent from tokenizer config {config}; the template read from {source}",
    remedy: REFETCH,
    test: "tests/chat/template.sh::a_template_using_an_undeclared_token_is_refused",
};

pub static UNPARSABLE: Cause = Cause {
    id: "template-unparsable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "compiling the chat template",
    cause: "the template does not parse: it is not Jinja the template engine reads, so a tag, expression or filter in it is malformed",
    observed: "line {line} of the template; the engine answered: {error}",
    remedy: REFETCH,
    test: "tests/chat/template.sh::a_template_that_does_not_parse_is_refused",
};

pub static RENDER_REFUSED: Cause = Cause {
    id: "render-refused",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "rendering {count} turn(s) through the chat template",
    cause: "the template failed to render them: the template itself refuses these turns (its raise_exception), as a template does for an order of roles it does not accept",
    observed: "roles: {roles}; the template said: {error}",
    remedy: "give the turns the template accepts (its raise_exception says which), or run with --chat-template off",
    test: "tests/chat/template.sh::a_template_that_raises_is_refused",
};

pub static RENDER_FAILED: Cause = Cause {
    id: "render-failed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "rendering {count} turn(s) through the chat template",
    cause: "the template failed to render them: it calls a method or function, or reads a value, that Ster's renderer does not provide",
    observed: "roles: {roles}; error kind {kind} at line {line}; the engine answered: {error}",
    remedy: "report the checkpoint at https://github.com/wisent-ai/ster/issues so the renderer learns what its template calls, and meanwhile run with --chat-template off",
    test: "tests/chat/template.sh::a_template_calling_an_unknown_method_is_refused",
};

pub static BOUNDARY_ABSENT: Cause = Cause {
    id: "boundary-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "locating where the assistant's turn begins",
    cause: "the template does not render the assistant's turn after the prompt, so the completion's tokens cannot be told from the prompt's",
    observed: "the prompt renders to {head} bytes and the conversation with the completion to {full}; they agree on the first {agreed}, and what follows (\"{after}\") does not hold the completion (\"{completion}\"), trimmed or not",
    remedy: "train with --chat-template off, or use the checkpoint's own template",
    test: "tests/chat/template.sh::a_template_that_rewrites_the_prompt_is_refused",
};

/// A checkpoint's chat template, read, compiled and rendered.
pub static TEMPLATE: Operation = Operation {
    id: "ster.chat.template",
    entries: &[],
    uses: &[&TOKENIZER],
    summary: "read a checkpoint's chat template from chat_template.jinja or its tokenizer config, compile it, and render turns through it when the run applies it",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &MODE_UNKNOWN,
        &FORM_UNKNOWN,
        &FORM_UNCHOSEN,
        &FORM_EMPTY,
        &TOKEN_UNDECLARED,
        &UNPARSABLE,
        &RENDER_REFUSED,
        &RENDER_FAILED,
        &BOUNDARY_ABSENT,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&TEMPLATE];
