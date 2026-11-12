//! Why one row of a JSONL export cannot become a labelled decision
//! (src/decide/pipeline/import/row.rs). `ster decisions import` lists every
//! refused row in its report under its cause, and refuses an export no row
//! of which is kept with the first refused row's cause, so its fix is the
//! row's own.

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/decide-pipeline.md, where this layer is listed.
const DOCS: &str = page!("decide-pipeline");

macro_rules! row_cause {
    ($name:ident, $id:literal, $cause:expr, $observed:expr, $remedy:expr, $test:literal) => {
        pub static $name: Cause = Cause {
            id: $id,
            code: Code::Config,
            usage: false,
            exit: None,
            failed: "importing line {line} of {path}",
            cause: $cause,
            observed: $observed,
            remedy: $remedy,
            test: concat!("tests/decide/import-rows.sh::", $test),
        };
    };
}

row_cause!(NOT_JSON, "row-not-json",
    "line {line} is not JSON, and every line of the export is one {\"context\", \"options\", \"label\"} object",
    "line {line}: {excerpt}; the JSON reader stopped at column {column}: {error}",
    "write line {line} as one JSON object on one line, such as {\"context\": \"the sea is flat\", \"options\": [\"calm\", \"rough\"], \"label\": 0}",
    "a_line_that_is_not_json_is_refused");
row_cause!(NOT_OBJECT, "row-not-object",
    "line {line} is JSON, {found}, where a row is one object holding context, options and label",
    "line {line}: {excerpt}",
    "write line {line} as one object, such as {\"context\": \"the sea is flat\", \"options\": [\"calm\", \"rough\"], \"label\": 0}",
    "a_line_that_is_not_an_object_is_refused");
row_cause!(FIELD_MISSING, "row-field-missing",
    "line {line} has no {field}, and a row states its context, its options and the index of the right one",
    "the fields it has: {fields}",
    "add {field} to line {line}, as in {\"context\": \"the sea is flat\", \"options\": [\"calm\", \"rough\"], \"label\": 0}",
    "a_row_without_a_field_is_refused");
row_cause!(CONTEXT_NOT_TEXT, "row-context-not-text",
    "the context of line {line} is {found}, where it is the text the question is asked about",
    "line {line}: {excerpt}",
    "write the context of line {line} as text in double quotes",
    "a_context_that_is_not_text_is_refused");
row_cause!(OPTIONS_NOT_LIST, "row-options-not-list",
    "the options of line {line} are {found}, where they are a list of the texts to choose from",
    "line {line}: {excerpt}",
    "write the options of line {line} as a list of two or more texts, such as [\"calm\", \"rough\"]",
    "options_that_are_not_a_list_are_refused");
row_cause!(OPTION_NOT_TEXT, "row-option-not-text",
    "option {index} of line {line} is {found}, where every option is text",
    "the options: {options}",
    "write option {index} of line {line} as text in double quotes",
    "an_option_that_is_not_text_is_refused");
row_cause!(LABEL_NOT_INDEX, "row-label-not-index",
    "the label of line {line} is {found}, where it is the index of the right option, a whole number from 0",
    "the options: {options}",
    "set the label of line {line} to the index of the right option, counted from 0",
    "a_label_that_is_not_an_index_is_refused");
row_cause!(CONTEXT_BLANK, "row-context-blank",
    "the context of line {line} is blank, so there is nothing to ask the question about",
    "context: {context}",
    "write the text the question is asked about as the context of line {line}, or remove the line",
    "a_blank_context_is_refused");
row_cause!(OPTIONS_FEW, "row-options-few",
    "line {line} holds too few options to choose between, {count}, and a choice needs at least two",
    "the options: {options}",
    "give line {line} at least two options, the right one among them",
    "a_row_with_one_option_is_refused");
row_cause!(OPTIONS_MANY, "row-options-many",
    "line {line} has {count} options, and Ster labels at most {max}, one letter each",
    "the options: {options}",
    "keep at most {max} options on line {line}, the right one among them",
    "a_row_with_too_many_options_is_refused");
row_cause!(OPTION_BLANK, "row-option-blank",
    "option {index} of line {line} is blank, and a blank option cannot be told from no option",
    "the options: {options}",
    "write option {index} of line {line}, or take it out and correct the label",
    "a_blank_option_is_refused");
row_cause!(OPTIONS_REPEATED, "row-options-repeated",
    "options {first} and {second} of line {line} carry the same text once trimmed, {text}, so the label cannot say which is right",
    "the options: {options}",
    "make options {first} and {second} of line {line} differ, or keep one of them and correct the label",
    "repeated_options_are_refused");
row_cause!(LABEL_OUTSIDE, "row-label-outside",
    "the label {label} of line {line} names no option: the row has {count} options, numbered from 0 to {last}",
    "the options: {options}",
    "set the label of line {line} to the index of the right option, 0 to {last}",
    "a_label_past_the_options_is_refused");

/// Why one row of a JSONL export is refused.
pub static ROWS: Operation = Operation {
    id: "ster.decisions.import.rows",
    entries: &[],
    uses: &[],
    summary: "read one {\"context\", \"options\", \"label\"} line of a JSONL export into one labelled choice, or name why it cannot be one",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &NOT_JSON,
        &NOT_OBJECT,
        &FIELD_MISSING,
        &CONTEXT_NOT_TEXT,
        &OPTIONS_NOT_LIST,
        &OPTION_NOT_TEXT,
        &LABEL_NOT_INDEX,
        &CONTEXT_BLANK,
        &OPTIONS_FEW,
        &OPTIONS_MANY,
        &OPTION_BLANK,
        &OPTIONS_REPEATED,
        &LABEL_OUTSIDE,
    ],
};
