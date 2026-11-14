# How reading a benchmark export fails

Every declared way reading the export `ster pairs import` names fails: the
file itself, the CSV records TruthfulQA, Do-Not-Answer and BiPO publish,
multiple-choice JSON rows, and LiveCodeBench's problems and examples. The
import's own flags are on [pairs-import.md](pairs-import.md). How a refusal
reads and which status it exits with is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/pairs-export.md`;
`tests/errors/catalogue.sh` refuses a page that differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.pairs.export`

read a benchmark export: the file, its CSV records, its multiple-choice JSON rows, or LiveCodeBench's problems and examples

Started through: run only by the functionalities that use it. Uses: `ster.file`. Failure catalogue `ster.pairs.export`: 8 declared cause(s) of its own.

<a id="ster-pairs-export-export-missing"></a>

### `ster.pairs.export.export-missing`

- **What failed:** reading the export {path}
- **Cause:** no file is there: nothing at {path} is the export {flag} names
- **Observed state shown:** {flag} {path}; its directory {parent}; the system answered: {error}
- **Fix:** name the downloaded export with {flag}; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_missing_export_is_refused`

<a id="ster-pairs-export-csv-no-header"></a>

### `ster.pairs.export.csv-no-header`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} holds no header row
- **Observed state shown:** the file holds no record: {bytes} bytes, none of them a field
- **Fix:** name the {benchmark} export, whose first record names its columns: {required}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::an_empty_csv_is_refused`

<a id="ster-pairs-export-csv-column-missing"></a>

### `ster.pairs.export.csv-column-missing`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} has no column "{column}", so it is not the {benchmark} export
- **Observed state shown:** the header names {columns}
- **Fix:** name the {benchmark} export, whose header names {required}, or the benchmark this export belongs to with --benchmark
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_csv_of_another_export_is_refused`

<a id="ster-pairs-export-csv-record-width"></a>

### `ster.pairs.export.csv-record-width`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} record {number} has {fields} fields where the header names {width}
- **Observed state shown:** the header: {header}
- **Fix:** quote every field of record {number} that holds a comma, or name the {benchmark} export as it was published
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_csv_record_of_another_width_is_refused`

<a id="ster-pairs-export-csv-quote-open"></a>

### `ster.pairs.export.csv-quote-open`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} ends inside a quoted field
- **Observed state shown:** a field opened with a quote is never closed before the end of the file: the quote opens on line {line} of {lines}, in record {record}, and its text starts "{start}"
- **Fix:** close the quoted field, doubling every quote inside it, or name the {benchmark} export as it was published
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_csv_ending_inside_quotes_is_refused`

<a id="ster-pairs-export-row-not-json"></a>

### `ster.pairs.export.row-not-json`

- **What failed:** reading the multiple-choice rows of {path}
- **Cause:** {path} line {line} is not JSON, and an export that does not open with [ is JSON Lines, one JSON row object on each non-blank line
- **Observed state shown:** line {line}: {excerpt}; the JSON reader stopped at column {column}: {error}
- **Fix:** write line {line} as one JSON row object, as every non-blank line of JSON Lines is
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_json_lines_row_that_is_not_json_is_refused`

<a id="ster-pairs-export-row-unclosed"></a>

### `ster.pairs.export.row-unclosed`

- **What failed:** reading the multiple-choice rows of {path}
- **Cause:** the JSON value line {line} of {path} opens does not close on that line: the export is JSON spread over several lines (printed with indentation), and one that does not open with [ is JSON Lines, one row object per line
- **Observed state shown:** line {line}: {excerpt}; the JSON reader ran out at column {column}: {error}
- **Fix:** write each row on one line (`jq -c . {path}` prints every value of the file on a line of its own), or write the rows as one JSON list opening with [
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::a_json_lines_row_that_is_not_json_is_refused`

<a id="ster-pairs-export-examples-no-questions"></a>

### `ster.pairs.export.examples-no-questions`

- **What failed:** reading the LiveCodeBench examples {path}
- **Cause:** {path} has no questions list
- **Observed state shown:** its top level: {top}
- **Fix:** name LiveCodeBench's question_examples.json with --examples: an object whose "questions" lists each question's good and bad example
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/imports/import.sh::examples_without_a_questions_list_are_refused`
