# How a pair set fails

Every declared way reading, checking and writing a pair-set file fails (the
layer every command that takes a pair set uses), and how adding, removing,
editing, saving, merging, splitting, sampling and inspecting a set fail. Importing a benchmark
export is on [pairs-import.md](pairs-import.md) and reading the export on
[pairs-export.md](pairs-export.md). How a refusal reads and which status it
exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/pairs.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.pairs.file`

read a pair-set file, check that every pair has both sides, and write one back

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.pairs.file`: 3 declared cause(s) of its own.

<a id="ster-pairs-file-pair-set-missing"></a>

### `ster.pairs.file.pair-set-missing`

- **What failed:** reading pair set {path}
- **Cause:** no file is there: nothing at {path} is the pair set named
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name an existing pair-set file (`ls -ln {parent}` shows what that directory holds), or write one with `ster pairs add --pairs {path} --positive <text> --negative <text>`
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_missing_pair_set_is_refused`

<a id="ster-pairs-file-pair-set-empty"></a>

### `ster.pairs.file.pair-set-empty`

- **What failed:** checking pair set {label}
- **Cause:** pair set {label} contains no pairs
- **Observed state shown:** pairs: 0; trait_name: "{trait}"
- **Fix:** add a pair with `ster pairs add --pairs <file> --positive <text> --negative <text>`, or name a set that holds pairs
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::removing_the_last_pair_is_refused_and_the_file_kept`

<a id="ster-pairs-file-pair-set-blank-prompt"></a>

### `ster.pairs.file.pair-set-blank-prompt`

- **What failed:** checking pair set {label}
- **Cause:** pair set {label} contains an empty positive or negative prompt
- **Observed state shown:** pair {index} has an empty {side}
- **Fix:** write that side with `ster pairs edit --pairs <file> --index {index} --{side} <text>`, or take the pair out with `ster pairs remove --pairs <file> --index {index}`
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/file.sh::a_pair_with_an_empty_side_is_refused`

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
- **Observed state shown:** none of --positive, --negative or --trait is given; {path} ({state}) is left as it is
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

## `ster.pairs.output`

check, before any set is read, that the path a drawn pair set is written to holds no file, so a set already there is never replaced

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.pairs.output`: 1 declared cause(s) of its own.

<a id="ster-pairs-output-output-exists"></a>

### `ster.pairs.output.output-exists`

- **What failed:** choosing --{flag} {path}
- **Cause:** {path} already holds a file, and a drawn pair set is written only where none is; nothing was written
- **Observed state shown:** --{flag} {path} ({state})
- **Fix:** give --{flag} a path where no file is, or move {path} away first
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/draw.sh::an_output_that_exists_is_refused`

## `ster.pairs.split`

part a pair set into a training set and a held-out set by a shuffle the seed fixes, each side keeping the set's order, and write both

Started through: `ster pairs split`. Uses: `ster.pairs.output`, `ster.vector.holdout`, `ster.pairs.file`. Failure catalogue `ster.pairs.split`: 1 declared cause(s) of its own.

<a id="ster-pairs-split-outputs-same"></a>

### `ster.pairs.split.outputs-same`

- **What failed:** choosing where pairs split writes
- **Cause:** --train and --test both name {path}, and the two sides are two files
- **Observed state shown:** --train {path} --test {path}
- **Fix:** give --train and --test two different paths
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/draw.sh::one_path_for_both_sides_is_refused`

## `ster.pairs.sample`

draw a number of pairs out of a pair set by a shuffle the seed fixes, keeping the set's order, and write them

Started through: `ster pairs sample`. Uses: `ster.pairs.output`, `ster.pairs.file`. Failure catalogue `ster.pairs.sample`: 1 declared cause(s) of its own.

<a id="ster-pairs-sample-count-outside"></a>

### `ster.pairs.sample.count-outside`

- **What failed:** drawing {count} pairs from {path}
- **Cause:** --count {count} is not between 1 and the {total} pair(s) the set holds, and a sample repeats no pair
- **Observed state shown:** --count {count}; {path} holds {total} pair(s)
- **Fix:** give --count a number from 1 to {total}
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/draw.sh::a_count_past_the_set_is_refused`

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
- **Test:** `tests/pairs/imports/inspect.sh::a_band_count_that_does_not_divide_64_is_refused`

<a id="ster-pairs-dedupe-threshold-too-wide"></a>

### `ster.pairs.dedupe.threshold-too-wide`

- **What failed:** checking the near-duplicate settings
- **Cause:** dedupe threshold {bits} exceeds the 64-bit simhash width
- **Observed state shown:** --dedupe-bits {bits}
- **Fix:** give --dedupe-bits a Hamming distance from 0 to 64
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/pairs/imports/inspect.sh::a_threshold_wider_than_the_fingerprint_is_refused`

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
- **Test:** `tests/pairs/imports/inspect.sh::a_ratio_below_one_is_refused`
