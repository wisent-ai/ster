//! How reading a benchmark export fails: the file itself, the CSV records
//! TruthfulQA, Do-Not-Answer and BiPO publish, multiple-choice JSON rows, and
//! LiveCodeBench's problems and examples.

use std::path::Path;

use wisent_errors::{Cause, Code, Operation};

use super::super::{file, refuse};

/// docs/guide/errors/pairs-export.md, where this functionality is listed.
const DOCS: &str = page!("pairs-export");

/// The refusal an export that cannot be read as text meets: nothing there
/// (this functionality's own cause, naming `flag`, the flag that named it),
/// or the `ster.file` layer's reason it cannot be read as UTF-8 text.
pub fn read_refusal(path: &Path, flag: &str, error: &std::io::Error) -> anyhow::Error {
    if error.kind() != std::io::ErrorKind::NotFound {
        return file::read::refusal(&format!("the export named by {flag}"), path, error);
    }
    refuse(
        &EXPORT_MISSING,
        &[
            ("path", &path.display().to_string()),
            ("flag", flag),
            ("parent", &file::parent(path).display().to_string()),
            ("error", &error.to_string()),
        ],
    )
}

pub static EXPORT_MISSING: Cause = Cause {
    id: "export-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading the export {path}",
    cause: "no file is there: nothing at {path} is the export {flag} names",
    observed: "{flag} {path}; its directory {parent}; the system answered: {error}",
    remedy: "name the downloaded export with {flag}; `ls -ln {parent}` shows what that directory holds",
    test: "tests/pairs/imports/import.sh::a_missing_export_is_refused",
};

pub static CSV_NO_HEADER: Cause = Cause {
    id: "csv-no-header",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the CSV export {path}",
    cause: "{path} holds no header row",
    observed: "the file holds no record: {bytes} bytes, none of them a field",
    remedy: "name the {benchmark} export, whose first record names its columns: {required}",
    test: "tests/pairs/imports/import.sh::an_empty_csv_is_refused",
};

pub static CSV_COLUMN_MISSING: Cause = Cause {
    id: "csv-column-missing",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the CSV export {path}",
    cause: "{path} has no column \"{column}\", so it is not the {benchmark} export",
    observed: "the header names {columns}",
    remedy: "name the {benchmark} export, whose header names {required}, or the benchmark this export belongs to with --benchmark",
    test: "tests/pairs/imports/import.sh::a_csv_of_another_export_is_refused",
};

pub static CSV_RECORD_WIDTH: Cause = Cause {
    id: "csv-record-width",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the CSV export {path}",
    cause: "{path} record {number} has {fields} fields where the header names {width}",
    observed: "the header: {header}",
    remedy: "quote every field of record {number} that holds a comma, or name the {benchmark} export as it was published",
    test: "tests/pairs/imports/import.sh::a_csv_record_of_another_width_is_refused",
};

pub static CSV_QUOTE_OPEN: Cause = Cause {
    id: "csv-quote-open",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the CSV export {path}",
    cause: "{path} ends inside a quoted field",
    observed: "a field opened with a quote is never closed before the end of the file: the quote opens on line {line} of {lines}, in record {record}, and its text starts \"{start}\"",
    remedy: "close the quoted field, doubling every quote inside it, or name the {benchmark} export as it was published",
    test: "tests/pairs/imports/import.sh::a_csv_ending_inside_quotes_is_refused",
};

pub static ROW_NOT_JSON: Cause = Cause {
    id: "row-not-json",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the multiple-choice rows of {path}",
    cause: "{path} line {line} is not JSON, and an export that does not open with [ is JSON Lines, one JSON row object on each non-blank line",
    observed: "line {line}: {excerpt}; the JSON reader stopped at column {column}: {error}",
    remedy: "write line {line} as one JSON row object, as every non-blank line of JSON Lines is",
    test: "tests/pairs/imports/import.sh::a_json_lines_row_that_is_not_json_is_refused",
};

pub static ROW_UNCLOSED: Cause = Cause {
    id: "row-unclosed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the multiple-choice rows of {path}",
    cause: "the JSON value line {line} of {path} opens does not close on that line: the export is JSON spread over several lines (printed with indentation), and one that does not open with [ is JSON Lines, one row object per line",
    observed: "line {line}: {excerpt}; the JSON reader ran out at column {column}: {error}",
    remedy: "write each row on one line (`jq -c . {path}` prints every value of the file on a line of its own), or write the rows as one JSON list opening with [",
    test: "tests/pairs/imports/import.sh::a_json_lines_row_that_is_not_json_is_refused",
};

pub static EXAMPLES_NO_QUESTIONS: Cause = Cause {
    id: "examples-no-questions",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the LiveCodeBench examples {path}",
    cause: "{path} has no questions list",
    observed: "its top level: {top}",
    remedy: "name LiveCodeBench's question_examples.json with --examples: an object whose \"questions\" lists each question's good and bad example",
    test: "tests/pairs/imports/import.sh::examples_without_a_questions_list_are_refused",
};

/// Reading one benchmark export into rows.
pub static EXPORT: Operation = Operation {
    id: "ster.pairs.export",
    entries: &[],
    uses: &[&file::FILE],
    summary: "read a benchmark export: the file, its CSV records, its multiple-choice JSON rows, or LiveCodeBench's problems and examples",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &EXPORT_MISSING,
        &CSV_NO_HEADER,
        &CSV_COLUMN_MISSING,
        &CSV_RECORD_WIDTH,
        &CSV_QUOTE_OPEN,
        &ROW_NOT_JSON,
        &ROW_UNCLOSED,
        &EXAMPLES_NO_QUESTIONS,
    ],
};
