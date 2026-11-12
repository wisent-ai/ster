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
