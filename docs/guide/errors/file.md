# How reading and writing one of Ster's files fails

Every declared way reading a file, reading a JSON document or writing,
linking or removing a file fails, whichever role the file plays: a pair set, a
steering artifact, a calibration, a checkpoint's state, a workspace record.
The functionality that reads the file names it (`reading the pair set …`) and
keeps its own cause for a file that is not there, since what to name instead
depends on what the file is; every other reason is one cause here, split by
the reason the system or the JSON reader gives. How a refusal reads and which
status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/file.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
