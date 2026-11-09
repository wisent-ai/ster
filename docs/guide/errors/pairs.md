# How a pair-set file fails

Every declared way reading, checking and writing a pair-set file fails: the
layer every command that takes a pair set uses. How a refusal reads and which
status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/pairs.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.pairs.file`

read a pair-set file, check that every pair has both sides, and write one back

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.pairs.file`: 7 declared cause(s) of its own.

<a id="ster-pairs-file-pair-set-missing"></a>

### `ster.pairs.file.pair-set-missing`

- **What failed:** reading pair set {path}
- **Cause:** no file is there: {error}
- **Observed state shown:** path: {path}
- **Fix:** name an existing pair-set file, or write one with `ster pairs add --pairs {path} --positive <text> --negative <text>`
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/file.sh::a_missing_pair_set_is_refused`

<a id="ster-pairs-file-pair-set-unreadable"></a>

### `ster.pairs.file.pair-set-unreadable`

- **What failed:** reading pair set {path}
- **Cause:** the file cannot be read: {error}
- **Observed state shown:** path: {path}
- **Fix:** name a pair-set file this account can read
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/file.sh::a_directory_named_as_a_pair_set_is_refused`

<a id="ster-pairs-file-pair-set-malformed"></a>

### `ster.pairs.file.pair-set-malformed`

- **What failed:** reading pair set {path}
- **Cause:** the file is not a pair-set document
- **Observed state shown:** invalid pair set JSON in {path}: {error}
- **Fix:** write it as {"trait_name": "<trait>", "pairs": [{"positive": "<text>", "negative": "<text>"}]}, as docs/examples/pairs.json is
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/file.sh::a_file_that_is_not_a_pair_set_is_refused`

<a id="ster-pairs-file-pair-set-empty"></a>

### `ster.pairs.file.pair-set-empty`

- **What failed:** checking pair set {label}
- **Cause:** pair set {label} contains no pairs
- **Observed state shown:** pairs: 0
- **Fix:** add a pair with `ster pairs add --pairs <file> --positive <text> --negative <text>`, or name a set that holds pairs
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/file.sh::removing_the_last_pair_is_refused_and_the_file_kept`

<a id="ster-pairs-file-pair-set-blank-prompt"></a>

### `ster.pairs.file.pair-set-blank-prompt`

- **What failed:** checking pair set {label}
- **Cause:** pair set {label} contains an empty positive or negative prompt
- **Observed state shown:** pair {index} has an empty {side}
- **Fix:** write that side with `ster pairs edit --pairs <file> --index {index} --{side} <text>`, or take the pair out with `ster pairs remove --pairs <file> --index {index}`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/file.sh::a_pair_with_an_empty_side_is_refused`

<a id="ster-pairs-file-pair-set-directory"></a>

### `ster.pairs.file.pair-set-directory`

- **What failed:** writing pair set {path}
- **Cause:** its directory {directory} cannot be created: {error}
- **Observed state shown:** path: {path}
- **Fix:** name a path whose directory this account can create, or create {directory} first
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/file.sh::a_set_under_a_file_is_refused`

<a id="ster-pairs-file-pair-set-unwritable"></a>

### `ster.pairs.file.pair-set-unwritable`

- **What failed:** writing pair set {path}
- **Cause:** the file cannot be written: {error}
- **Observed state shown:** path: {path}
- **Fix:** make its directory writable for this account, or name another path
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/file.sh::a_set_in_a_read_only_directory_is_refused`
