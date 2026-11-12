# How a pair set fails

Every declared way reading, checking and writing a pair-set file fails (the
layer every command that takes a pair set uses), and how adding, removing,
editing, saving, merging and inspecting a set fail. Importing a benchmark
export is on [pairs-import.md](pairs-import.md) and reading the export on
[pairs-export.md](pairs-export.md). How a refusal reads and which status it
exits with is on [How Ster fails](../errors.md).

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

## `ster.pairs.index`

find one pair of a set by the zero-based index `ster pairs remove` and `ster pairs edit` take

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.pairs.index`: 1 declared cause(s) of its own.

<a id="ster-pairs-index-index-outside"></a>

### `ster.pairs.index.index-outside`

- **What failed:** finding pair {index} of {path}
- **Cause:** pair index {index} is outside the set: {path} holds {count} pair(s), indexed from 0
- **Observed state shown:** pairs: {count}, the last at index {last}
- **Fix:** name an index from 0 to {last}; the pairs of {path} are numbered in the order the file lists them
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/author.sh::an_index_past_the_end_is_refused`

## `ster.pairs.side`

check the text given for one side of a pair before any set is read

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.pairs.side`: 1 declared cause(s) of its own.

<a id="ster-pairs-side-side-empty"></a>

### `ster.pairs.side.side-empty`

- **What failed:** writing the {side} side of a pair in {path}
- **Cause:** --{side} is empty or only whitespace, and every pair needs text on both sides
- **Observed state shown:** --{side} "{value}"
- **Fix:** give --{side} the text of that side
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/author.sh::an_empty_side_is_refused_before_the_set_is_read`

## `ster.pairs.add`

append one pair to a set, creating the file when nothing is there

Started through: `ster pairs add`. Uses: `ster.pairs.side`, `ster.pairs.file`. Failure catalogue `ster.pairs.add`: 0 declared cause(s) of its own.

## `ster.pairs.remove`

take one pair out of a set by its index; the last pair stays, since a set without pairs is refused

Started through: `ster pairs remove`. Uses: `ster.pairs.index`, `ster.pairs.file`. Failure catalogue `ster.pairs.remove`: 0 declared cause(s) of its own.

## `ster.pairs.edit`

change one pair's sides, or the set's trait name, in place

Started through: `ster pairs edit`. Uses: `ster.pairs.side`, `ster.pairs.index`, `ster.pairs.file`. Failure catalogue `ster.pairs.edit`: 1 declared cause(s) of its own.

<a id="ster-pairs-edit-nothing-to-change"></a>

### `ster.pairs.edit.nothing-to-change`

- **What failed:** editing pair {index} of {path}
- **Cause:** pairs edit changes nothing without --positive, --negative or --trait
- **Observed state shown:** none of --positive, --negative or --trait is given
- **Fix:** name what changes: --positive <text>, --negative <text> or --trait <name>
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/author.sh::an_edit_naming_no_change_is_refused`

## `ster.pairs.save`

write the pair set the desktop editor holds, checked as every loaded set is

Started through: `ster request pairs/save`. Uses: `ster.pairs.file`. Failure catalogue `ster.pairs.save`: 0 declared cause(s) of its own.

## `ster.pairs.merge`

write one pair set out of several, every pair of each in the order the sets are named, under one trait name

Started through: `ster pairs merge`, `ster request pairs/merge`. Uses: `ster.pairs.file`. Failure catalogue `ster.pairs.merge`: 2 declared cause(s) of its own.

<a id="ster-pairs-merge-too-few-sources"></a>

### `ster.pairs.merge.too-few-sources`

- **What failed:** merging pair sets
- **Cause:** pairs merge needs at least two pair sets and got {count}
- **Observed state shown:** pair sets named: {sources}
- **Fix:** name every set to merge with its own --pairs, two or more: one set is already itself
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/merge.sh::fewer_than_two_sets_are_refused_before_any_is_read`

<a id="ster-pairs-merge-trait-blank"></a>

### `ster.pairs.merge.trait-blank`

- **What failed:** merging pair sets
- **Cause:** pairs merge needs the trait name the merged set is fitted for
- **Observed state shown:** --trait "{trait}"
- **Fix:** give --trait the name of the trait every merged set expresses
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/merge.sh::a_blank_trait_is_refused_before_any_set_is_read`

## `ster.pairs.dedupe`

check the near-duplicate settings a caller gives: the SimHash Hamming threshold and the band count

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.pairs.dedupe`: 2 declared cause(s) of its own.

<a id="ster-pairs-dedupe-bands-invalid"></a>

### `ster.pairs.dedupe.bands-invalid`

- **What failed:** checking the near-duplicate settings
- **Cause:** dedupe band count {bands} must divide the 64-bit simhash evenly, such as 4, 8, 16, or 32
- **Observed state shown:** --dedupe-bands {bands}
- **Fix:** give --dedupe-bands a divisor of 64: 1, 2, 4, 8, 16, 32 or 64
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/inspect.sh::a_band_count_that_does_not_divide_64_is_refused`

<a id="ster-pairs-dedupe-threshold-too-wide"></a>

### `ster.pairs.dedupe.threshold-too-wide`

- **What failed:** checking the near-duplicate settings
- **Cause:** dedupe threshold {bits} exceeds the 64-bit simhash width
- **Observed state shown:** --dedupe-bits {bits}
- **Fix:** give --dedupe-bits a Hamming distance from 0 to 64
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/inspect.sh::a_threshold_wider_than_the_fingerprint_is_refused`

## `ster.pairs.inspect`

audit a pair set without a model: near-duplicates, the length balance of each pair, and the diversity of its positives

Started through: `ster pairs inspect`, `ster request pairs/inspect`. Uses: `ster.pairs.dedupe`, `ster.workspace.active`, `ster.pairs.file`. Failure catalogue `ster.pairs.inspect`: 1 declared cause(s) of its own.

<a id="ster-pairs-inspect-ratio-invalid"></a>

### `ster.pairs.inspect.ratio-invalid`

- **What failed:** checking the length balance setting
- **Cause:** a pair's longer side over its shorter side is 1 or more, so a ratio below 1, or one that is not a number, counts every pair or none as unbalanced
- **Observed state shown:** --unbalanced-ratio {ratio}
- **Fix:** give --unbalanced-ratio a number of 1 or more: 3 counts a pair whose longer side has over three times the characters of the other
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/inspect.sh::a_ratio_below_one_is_refused`
