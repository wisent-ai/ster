# How a checkpoint's chat template fails

Every declared way a checkpoint's chat template fails: choosing whether a
run applies it (`--chat-template auto|off`), reading it from
`chat_template.jinja` or the `chat_template` of `tokenizer_config.json`,
compiling it, and rendering turns through it. A template is read when the
model loads and refused only by a run that applies it; `--chat-template off`
encodes raw text whatever it says. Encoding the rendered text is on
[runtime.md](runtime.md). How a refusal reads and which status it exits with
is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/chat.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.chat.template`

read a checkpoint's chat template from chat_template.jinja or its tokenizer config, compile it, and render turns through it when the run applies it

Started through: run only by the functionalities that use it. Uses: `ster.model.tokenizer`. Failure catalogue `ster.chat.template`: 9 declared cause(s) of its own.

<a id="ster-chat-template-mode-unknown"></a>

### `ster.chat.template.mode-unknown`

- **What failed:** reading the chat template mode
- **Cause:** unknown chat template mode {value}; expected auto or off
- **Observed state shown:** --chat-template {value}
- **Fix:** give --chat-template auto or off
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::an_unknown_mode_is_refused`

<a id="ster-chat-template-template-form-unknown"></a>

### `ster.chat.template.template-form-unknown`

- **What failed:** reading the chat_template of the tokenizer config
- **Cause:** it is neither a template nor a list of them
- **Observed state shown:** chat_template: {value}; config: {path}
- **Fix:** fetch the checkpoint's tokenizer_config.json and chat_template.jinja again, or run with --chat-template off to encode raw text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_of_no_known_form_is_refused`

<a id="ster-chat-template-template-default-absent"></a>

### `ster.chat.template.template-default-absent`

- **What failed:** reading the chat_template of the tokenizer config
- **Cause:** it lists several named templates and none of them is the default, and only the default is the conversation format
- **Observed state shown:** names listed: {names}; config: {path}
- **Fix:** fetch the checkpoint's tokenizer_config.json and chat_template.jinja again, or run with --chat-template off to encode raw text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_list_without_a_default_is_refused`

<a id="ster-chat-template-template-entry-empty"></a>

### `ster.chat.template.template-entry-empty`

- **What failed:** reading the chat_template of the tokenizer config
- **Cause:** its chosen entry {name} holds no template string
- **Observed state shown:** entry: {entry}; config: {path}
- **Fix:** fetch the checkpoint's tokenizer_config.json and chat_template.jinja again, or run with --chat-template off to encode raw text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_entry_without_text_is_refused`

<a id="ster-chat-template-token-undeclared"></a>

### `ster.chat.template.token-undeclared`

- **What failed:** compiling the chat template
- **Cause:** the template writes {name}, but the tokenizer config declares none, so every rendering would lack it
- **Observed state shown:** {name}: absent from tokenizer config {config}; the template read from {source}
- **Fix:** fetch the checkpoint's tokenizer_config.json and chat_template.jinja again, or run with --chat-template off to encode raw text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_using_an_undeclared_token_is_refused`

<a id="ster-chat-template-template-unparsable"></a>

### `ster.chat.template.template-unparsable`

- **What failed:** compiling the chat template
- **Cause:** the template does not parse: it is not Jinja the template engine reads, so a tag, expression or filter in it is malformed
- **Observed state shown:** line {line} of the template; the engine answered: {error}
- **Fix:** fetch the checkpoint's tokenizer_config.json and chat_template.jinja again, or run with --chat-template off to encode raw text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_that_does_not_parse_is_refused`

<a id="ster-chat-template-render-refused"></a>

### `ster.chat.template.render-refused`

- **What failed:** rendering {count} turn(s) through the chat template
- **Cause:** the template failed to render them: the template itself refuses these turns (its raise_exception), as a template does for an order of roles it does not accept
- **Observed state shown:** roles: {roles}; the template said: {error}
- **Fix:** give the turns the template accepts (its raise_exception says which), or run with --chat-template off
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_that_raises_is_refused`

<a id="ster-chat-template-render-failed"></a>

### `ster.chat.template.render-failed`

- **What failed:** rendering {count} turn(s) through the chat template
- **Cause:** the template failed to render them: it calls a method or function, or reads a value, that Ster's renderer does not provide
- **Observed state shown:** roles: {roles}; error kind {kind} at line {line}; the engine answered: {error}
- **Fix:** report the checkpoint at https://github.com/wisent-ai/ster/issues so the renderer learns what its template calls, and meanwhile run with --chat-template off
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_calling_an_unknown_method_is_refused`

<a id="ster-chat-template-boundary-absent"></a>

### `ster.chat.template.boundary-absent`

- **What failed:** locating where the assistant's turn begins
- **Cause:** the template does not render the assistant's turn after the prompt, so the completion's tokens cannot be told from the prompt's
- **Observed state shown:** the prompt renders to {head} bytes and the conversation with the completion to {full}; they agree on the first {agreed}, and what follows ("{after}") does not hold the completion ("{completion}"), trimmed or not
- **Fix:** train with --chat-template off, or use the checkpoint's own template
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/chat/template.sh::a_template_that_rewrites_the_prompt_is_refused`
