//! How `ster decisions fetch` fails (src/decide/pipeline/fetch.rs): asking the
//! Hugging Face datasets server for a split's rows, reading its pages, and
//! turning a text column and a class-label column into labelled decisions.

use wisent_errors::{Cause, Code, Operation};

use super::super::file::FILE;
use super::pipeline::QUESTION;

/// docs/guide/errors/decide-pipeline.md, where this functionality is listed.
const DOCS: &str = page!("decide-pipeline");

macro_rules! fetch_cause {
    ($name:ident, $id:literal, $code:expr, $usage:literal, $failed:literal, $cause:literal, $observed:literal, $remedy:literal, $test:literal) => {
        pub static $name: Cause = Cause {
            id: $id,
            code: $code,
            usage: $usage,
            exit: None,
            failed: $failed,
            cause: $cause,
            observed: $observed,
            remedy: $remedy,
            test: concat!("tests/decide/fetch.sh::", $test),
        };
    };
}

fetch_cause!(COUNT_NONE, "count-none", Code::Config, true,
    "checking the fetch settings",
    "--count 0 asks for no row, so no labelled decision could be written",
    "--count 0",
    "give --count the number of rows to read, 1 or more",
    "a_count_of_zero_is_refused");
fetch_cause!(SERVER_UNREACHABLE, "server-unreachable", Code::InfraDown, false,
    "asking the datasets server for rows of {dataset}",
    "no answer came from https://datasets-server.huggingface.co: the connection was not made or broke (no network, a name the resolver does not know, a proxy or TLS failing)",
    "GET {url}: the client's error kind {kind}; it answered: {error}",
    "run the command again once https://datasets-server.huggingface.co answers from this machine",
    "an_unreachable_datasets_server_is_refused");
fetch_cause!(SERVER_REFUSED, "server-refused", Code::NotFound, false,
    "asking the datasets server for rows of {dataset}",
    "the datasets server answered {status}: it serves no rows of {dataset} with config {config} and split {split} to this account (a dataset, config or split it does not hold, or a gated one)",
    "GET {url} answered HTTP {status}; the server said: {said}",
    "name a dataset, --config and --split https://huggingface.co/datasets/{dataset} lists, or one this account may read",
    "an_unknown_dataset_is_refused");
fetch_cause!(SERVER_LIMITED, "server-limited", Code::RateLimit, false,
    "asking the datasets server for rows of {dataset}",
    "the datasets server answered 429: it limits how often it is asked, and this fetch passed the limit",
    "GET {url} answered HTTP {status}; the server said: {said}",
    "run the command again after a while, or fetch fewer rows with --count",
    "a_datasets_server_limiting_requests_is_refused");
fetch_cause!(SERVER_FAILING, "server-failing", Code::InfraDown, false,
    "asking the datasets server for rows of {dataset}",
    "the datasets server answered {status}: it failed on its side, so the request was right and the server could not serve it",
    "GET {url} answered HTTP {status}; the server said: {said}",
    "run the command again once https://status.huggingface.co shows the datasets server serving",
    "a_datasets_server_failing_on_its_side_is_refused");
fetch_cause!(PAGE_UNREAD, "page-unread", Code::InfraDown, false,
    "reading the datasets server's page of {dataset}",
    "the page broke off before it was read whole: the server closed the connection midway, or answered bytes that are not UTF-8 text",
    "GET {url} answered HTTP {status}; error kind {kind}; the client answered: {error}",
    "run the command again; the rows are read from the start",
    "a_page_that_breaks_off_is_refused");
fetch_cause!(PAGE_NOT_JSON, "page-not-json", Code::Refused, false,
    "reading the datasets server's page of {dataset}",
    "the server answered text that is not JSON, where the rows route answers one JSON document",
    "GET {url} answered HTTP 200 with {size} bytes: {page}; the JSON reader stopped at line {line}, column {column}: {error}",
    "run the command again; when it repeats, https://huggingface.co/docs/dataset-viewer says what the rows route answers now",
    "a_page_that_is_not_json_is_refused");
fetch_cause!(PAGE_ROWLESS, "page-rowless", Code::Refused, false,
    "reading the datasets server's page of {dataset} at offset {offset}",
    "the page carries no `rows` array, which the rows route answers with",
    "the page's fields: {fields}",
    "run the command again; when it repeats, https://huggingface.co/docs/dataset-viewer says what the rows route answers now",
    "a_page_without_rows_is_refused");
fetch_cause!(PAGE_FEATURELESS, "page-featureless", Code::Refused, false,
    "reading the columns of {dataset}",
    "the page carries no `features` array naming the dataset's columns, which the rows route answers with",
    "the page's fields: {fields}",
    "run the command again; when it repeats, https://huggingface.co/docs/dataset-viewer says what the rows route answers now",
    "a_page_without_features_is_refused");
fetch_cause!(LABEL_ABSENT, "label-column-absent", Code::Config, false,
    "reading the label column '{label}' of {dataset}",
    "the dataset has no column named '{label}'",
    "the dataset's columns: {columns}",
    "name one of the dataset's class-label columns with --label-field",
    "a_label_column_the_dataset_lacks_is_refused");
fetch_cause!(LABEL_NOT_CLASS, "label-not-class", Code::Config, false,
    "reading the label column '{label}' of {dataset}",
    "column '{label}' is no class label with named classes, so its values name no option",
    "column '{label}' is {feature}",
    "name a ClassLabel column with --label-field; the dataset's columns are listed on https://huggingface.co/datasets/{dataset}",
    "a_label_column_that_is_not_a_class_label_is_refused");
fetch_cause!(CLASS_NOT_TEXT, "class-name-not-text", Code::Refused, false,
    "reading the class names of column '{label}' of {dataset}",
    "column '{label}' names a class with something other than text, and an option is named by text",
    "class {index}: {found}",
    "name another class-label column with --label-field, or the dataset whose classes are named",
    "a_class_named_with_no_text_is_refused");
fetch_cause!(CLASSES_TOO_FEW, "classes-too-few", Code::Config, false,
    "reading the class names of column '{label}' of {dataset}",
    "column '{label}' names {count} class(es), and a choice needs at least two options",
    "classes: {classes}",
    "name a class-label column with two or more classes with --label-field",
    "a_label_column_with_one_class_is_refused");
fetch_cause!(CLASSES_TOO_MANY, "classes-too-many", Code::Config, false,
    "reading the class names of column '{label}' of {dataset}",
    "column '{label}' names {count} classes, and a choice is answered under one of {most} letters",
    "classes: {count}; the most: {most}",
    "name a class-label column with at most {most} classes with --label-field",
    "a_label_column_with_too_many_classes_is_refused");
fetch_cause!(TEXT_ABSENT, "text-column-absent", Code::Config, false,
    "reading row {index} of {dataset}",
    "the row has no column '{text}', so it carries no text to decide about",
    "the row's columns: {columns}",
    "name one of the dataset's text columns with --text-field",
    "a_text_column_the_dataset_lacks_is_refused");
fetch_cause!(TEXT_MISTYPED, "text-column-mistyped", Code::Config, false,
    "reading row {index} of {dataset}",
    "column '{text}' holds {found} in this row, not text",
    "row {index}: {text}: {found}",
    "name a column that holds text with --text-field",
    "a_text_column_that_holds_no_text_is_refused");
fetch_cause!(NONE_KEPT, "none-kept", Code::Config, false,
    "turning rows {first}..{end} of {dataset} into labelled decisions",
    "no row read carried both a text and a label the dataset's classes name",
    "rows read: {rows}; skipped: {skipped}; classes: {classes}",
    "fetch from another --offset or --split, or name the columns that hold the text and the label",
    "rows_without_a_known_label_are_refused");

/// Labelled decisions from a Hugging Face classification dataset.
pub static FETCH: Operation = Operation {
    id: "ster.decisions.fetch",
    entries: &["ster decisions fetch"],
    uses: &[&QUESTION, &FILE],
    summary: "read a split's rows from the Hugging Face datasets server and turn its text column and class-label column into one labelled choice question per row",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &COUNT_NONE, &SERVER_UNREACHABLE, &SERVER_REFUSED, &SERVER_LIMITED, &SERVER_FAILING,
        &PAGE_UNREAD, &PAGE_NOT_JSON, &PAGE_ROWLESS, &PAGE_FEATURELESS, &LABEL_ABSENT,
        &LABEL_NOT_CLASS, &CLASS_NOT_TEXT, &CLASSES_TOO_FEW, &CLASSES_TOO_MANY, &TEXT_ABSENT,
        &TEXT_MISTYPED, &NONE_KEPT,
    ],
};
