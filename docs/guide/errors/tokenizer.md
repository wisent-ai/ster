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
