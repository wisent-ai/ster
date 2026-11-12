# How the first-use walkthrough fails

Every declared way `ster onboarding` fails: finding where this account keeps
its progress, and reading back a progress file that belongs to another
journey or stands at a screen this journey does not hold. The state file
failing to be read, parsed or written is on [file.md](file.md); importing a
pair set with `--import-pairs` is on [workspace.md](workspace.md). How a
refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/onboarding.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
