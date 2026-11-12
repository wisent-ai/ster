# How a steering artifact fails

Every declared way reading, checking and writing a steering artifact fails:
the JSON document `ster vector train` and `ster tune bipo` write, one vector
per layer it steers. `ster vector inspect` reads one; every command that
takes `--vector` reads one the same way, and steering a model with it is on
[runtime.md](runtime.md). How a refusal reads and which status it exits with
is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/steering.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
