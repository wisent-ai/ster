# How training, scoring and rewriting adapters fail

Every declared way a `ster tune` command, `ster vector evaluate --strengths`
and `ster vector ablate` fail, with their `ster request` twins: the settings
every trainer and each objective takes, the example sets, what every run
shares once its model is loaded, its checkpoints and resuming from them, the
reward policy optimization scores with and the conversations it samples,
choosing a steering strength, and fitting a trained artifact to the
checkpoint a merge, an export or an ablation writes. The adapter artifact
itself is on [lora.md](lora.md), loading the model on
[model.md](model.md), and encoding text on [runtime.md](runtime.md). How a
refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/tune.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
