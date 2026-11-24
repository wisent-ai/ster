# How listing Ster's sources fails

Every declared way `ster sources` fails: it lists the data other tools keep
that Ster reads, one adapter each, named for its source, and reads one file as
its adapter's command would. A file that is not that source is answered with
why, so the refusals are an adapter Ster does not have, an adapter named by the
import plan's requirement number it was first listed under, and an answer to
`sources share` that cannot be kept. How a refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/sources.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.sources`

list the data other tools keep that Ster reads, one adapter each, named for its source, and say how many examples or pairs one file gives

Started through: `ster sources`, `ster sources discover`, `ster sources share`. Uses: none. Failure catalogue `ster.sources`: 3 declared cause(s) of its own.

<a id="ster-sources-sources-adapter-unknown"></a>

### `ster.sources.sources-adapter-unknown`

- **What failed:** discovering {path} as {adapter}
- **Cause:** {adapter} is not an adapter of ster
- **Observed state shown:** its adapters are {known}
- **Fix:** name one of the adapters `ster sources` lists
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/sources/discover.sh::an_unknown_adapter_is_refused`

<a id="ster-sources-sources-adapter-renamed"></a>

### `ster.sources.sources-adapter-renamed`

- **What failed:** discovering {path} as {adapter}
- **Cause:** {adapter} is the import plan's requirement number of an adapter, and ster's adapters are named, not numbered
- **Observed state shown:** {number} stands for the adapter named {name}
- **Fix:** name it {name}: `ster sources discover {name} --path {path}`
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/sources/discover.sh::an_adapter_named_by_its_requirement_number_is_refused_naming_its_name`

<a id="ster-sources-sources-share-unkept"></a>

### `ster.sources.sources-share-unkept`

- **What failed:** keeping whether discoveries report their counts
- **Cause:** {why}
- **Observed state shown:** the answer was not kept
- **Fix:** set XDG_CONFIG_HOME to a directory this user can write, then answer again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/sources/share.sh::an_answer_to_sharing_that_cannot_be_kept_is_refused`
