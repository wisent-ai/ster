# How LoRA adapters fail

Every declared way the rank, alpha, targets and layers of a training run's
adapters are refused, and every way reading, checking and writing a trained
adapter's artifact fails: its safetensors weights, the JSON sidecar beside
them, and the tensors the sidecar says it holds. `ster tune inspect` reads
one; every command that takes `--adapter` reads one the same way. How a
refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/lora.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
