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

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.pairs.export`: 11 declared cause(s) of its own.

<a id="ster-pairs-export-export-missing"></a>

### `ster.pairs.export.export-missing`

- **What failed:** reading the export {path}
- **Cause:** no file is there: {error}
- **Observed state shown:** {flag} {path}
- **Fix:** name the downloaded export with {flag}
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_missing_export_is_refused`

<a id="ster-pairs-export-export-unreadable"></a>

### `ster.pairs.export.export-unreadable`

- **What failed:** reading the export {path}
- **Cause:** it cannot be read as UTF-8 text: {error}
- **Observed state shown:** {flag} {path}
- **Fix:** name with {flag} a UTF-8 text file this account can read
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_directory_named_as_an_export_is_refused`

<a id="ster-pairs-export-csv-no-header"></a>

### `ster.pairs.export.csv-no-header`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} holds no header row
- **Observed state shown:** the file holds no record
- **Fix:** name the {benchmark} export, whose first record names its columns: {required}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::an_empty_csv_is_refused`

<a id="ster-pairs-export-csv-column-missing"></a>

### `ster.pairs.export.csv-column-missing`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} has no column "{column}", so it is not the {benchmark} export
- **Observed state shown:** the header names {columns}
- **Fix:** name the {benchmark} export, whose header names {required}, or the benchmark this export belongs to with --benchmark
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_csv_of_another_export_is_refused`

<a id="ster-pairs-export-csv-record-width"></a>

### `ster.pairs.export.csv-record-width`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} record {number} has {fields} fields where the header names {width}
- **Observed state shown:** the header: {header}
- **Fix:** quote every field of record {number} that holds a comma, or name the {benchmark} export as it was published
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_csv_record_of_another_width_is_refused`

<a id="ster-pairs-export-csv-quote-open"></a>

### `ster.pairs.export.csv-quote-open`

- **What failed:** reading the CSV export {path}
- **Cause:** {path} ends inside a quoted field
- **Observed state shown:** a field opened with a quote is never closed before the end of the file
- **Fix:** close the quoted field, doubling every quote inside it, or name the {benchmark} export as it was published
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_csv_ending_inside_quotes_is_refused`

<a id="ster-pairs-export-rows-not-a-list"></a>

### `ster.pairs.export.rows-not-a-list`

- **What failed:** reading the multiple-choice rows of {path}
- **Cause:** {path} is not a JSON list of rows
- **Observed state shown:** it starts with [ and the parser stopped: {error}
- **Fix:** write the rows as one JSON list of row objects, or as JSON Lines with one row object per line
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_broken_json_list_of_rows_is_refused`

<a id="ster-pairs-export-row-not-json"></a>

### `ster.pairs.export.row-not-json`

- **What failed:** reading the multiple-choice rows of {path}
- **Cause:** {path} line {line} is not JSON
- **Observed state shown:** the parser stopped: {error}
- **Fix:** write line {line} as one JSON row object, as every non-blank line of JSON Lines is
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::a_json_lines_row_that_is_not_json_is_refused`

<a id="ster-pairs-export-problems-not-a-list"></a>

### `ster.pairs.export.problems-not-a-list`

- **What failed:** reading the LiveCodeBench problems {path}
- **Cause:** {path} is not a JSON list of problems
- **Observed state shown:** the parser stopped: {error}
- **Fix:** name LiveCodeBench's problems.json with --source
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::problems_that_are_not_a_json_list_are_refused`

<a id="ster-pairs-export-examples-not-json"></a>

### `ster.pairs.export.examples-not-json`

- **What failed:** reading the LiveCodeBench examples {path}
- **Cause:** {path} is not JSON
- **Observed state shown:** the parser stopped: {error}
- **Fix:** name LiveCodeBench's question_examples.json with --examples
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::examples_that_are_not_json_are_refused`

<a id="ster-pairs-export-examples-no-questions"></a>

### `ster.pairs.export.examples-no-questions`

- **What failed:** reading the LiveCodeBench examples {path}
- **Cause:** {path} has no questions list
- **Observed state shown:** its top level: {top}
- **Fix:** name LiveCodeBench's question_examples.json with --examples: an object whose "questions" lists each question's good and bad example
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/pairs/import.sh::examples_without_a_questions_list_are_refused`
