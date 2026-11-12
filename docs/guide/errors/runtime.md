# How a loaded model's runtime fails

Every declared way a model fails once its weights are mapped: turning a
prompt, a training example, a preference side, a conversation or an answer
label into the token ids the checkpoint expects, through its tokenizer and,
when the run applies it, its chat template. Loading the model is on
[model.md](model.md). How a refusal reads and which status it exits with is
on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/runtime.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
