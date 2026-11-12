# How building and measuring labelled decision sets fails

Every declared way `ster decisions fetch`, `import`, `synthesize`, `split`
and `benchmark` fail: the question a fetched or imported set asks, what the
Hugging Face datasets server answers and the columns a dataset holds, the
rows of a JSONL export, a decision schema and what the writing model
answers, and the held-out fraction. Reading and writing their files is on
[file.md](file.md); reading a labelled set, a calibration and a model is on
[decide-calibration.md](decide-calibration.md) and
[model-load.md](model-load.md). How a refusal reads and which status it
exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/decide-pipeline.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
