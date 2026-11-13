# How listing Ster's sources fails

Every declared way `ster sources` fails: it lists the data other tools keep
that Ster reads, one adapter each, and reads one file as its adapter's
command would. A file that is not that source is answered with why, so the
one refusal is an adapter Ster does not have. How a refusal reads and which
status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/sources.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.sources`

list the data other tools keep that Ster reads, one adapter each (IMP-STER-<NN>), and say how many examples or pairs one file gives

Started through: `ster sources`. Uses: none. Failure catalogue `ster.sources`: 1 declared cause(s) of its own.

<a id="ster-sources-sources-adapter-unknown"></a>

### `ster.sources.sources-adapter-unknown`

- **What failed:** discovering {path} as {adapter}
- **Cause:** {adapter} is not an adapter of ster
- **Observed state shown:** its adapters are {known}
- **Fix:** name one of the adapters `ster sources` lists
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/sources/sources.sh::an_unknown_adapter_is_refused`
