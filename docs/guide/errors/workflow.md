# How the steering workflows fail

Every declared way the commands that read a model's states and fit or
measure directions fail: choosing the layers `--layers` names, reading a
prompt set, fitting a direction by `--method`, holding pairs out to rank on,
`vector train`, `optimize`, `curve` and `evaluate`, `vector compare`,
`vector project` and `vector score`, `extract` and `parity`. Reading a
steering artifact is on [steering.md](steering.md), loading a model on
[model-load.md](model-load.md), and how a refusal reads and which status it
exits with on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/workflow.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
