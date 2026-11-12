//! How building and measuring labelled decision sets fails
//! (src/decide/pipeline, src/cli/decisions): the question a fetched or
//! imported set asks, importing JSONL rows, synthesizing states through a
//! writing model, splitting a set, and benchmarking a model on one. Reading
//! and writing their files is the `ster.file` layer's.

use wisent_errors::{Cause, Code, Operation};

use super::super::file::FILE;
use super::super::inference::INFERENCE;
use super::super::model::weights::LOAD;
use super::rows::ROWS;
use super::{CALIBRATE, CALIBRATION, EXAMPLES, REQUEST};

/// docs/guide/errors/decide-pipeline.md, where these functionalities are listed.
const DOCS: &str = page!("decide-pipeline");

/// What a decision schema is, for the `ster.file` refusals that name it.
pub const SCHEMA_FORM: &str = "{\"domain\": \"<what the states are>\", \"questions\": {<id>: <question>}}, as docs/guide/decisions.md gives it";

macro_rules! pipeline_cause {
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
            test: $test,
        };
    };
}

pipeline_cause!(QUESTION_ID_BLANK, "question-id-blank", Code::Config, true,
    "checking the question every {source} example asks",
    "--question-id is blank, and every example answers its question under that id",
    "--question-id {value}",
    "give --question-id the id the question is answered under, such as sentiment",
    "tests/decide/pipeline.sh::a_blank_question_id_is_refused");
pipeline_cause!(INSTRUCTIONS_BLANK, "instructions-blank", Code::Config, true,
    "checking the question every {source} example asks",
    "--instructions is blank, and the model is asked the question in those words",
    "--instructions {value}",
    "give --instructions the question the model answers about each text",
    "tests/decide/pipeline.sh::blank_instructions_are_refused");

/// The one question every fetched or imported example asks.
pub static QUESTION: Operation = Operation {
    id: "ster.decisions.question",
    entries: &[],
    uses: &[],
    summary: "check the id and instructions of the one choice question a fetched or imported set asks, before any row is read",
    docs: DOCS,
    warnings: &[],
    causes: &[&QUESTION_ID_BLANK, &INSTRUCTIONS_BLANK],
};

pipeline_cause!(ROWS_NONE, "rows-none", Code::Config, false,
    "importing {path}",
    "the file holds no row: every line is blank",
    "path: {path} ({state})",
    "name the JSONL export with one {\"context\", \"options\", \"label\"} object per line",
    "tests/decide/pipeline.sh::an_export_without_rows_is_refused");

/// Labelled decisions from a JSONL export. A row it refuses is
/// `ster.decisions.import.rows`'s cause, in the report and, when no row is
/// kept, in the refusal.
pub static IMPORT: Operation = Operation {
    id: "ster.decisions.import",
    entries: &["ster decisions import"],
    uses: &[&QUESTION, &ROWS, &EXAMPLES, &FILE],
    summary: "read a JSONL export of context/options/label rows into one labelled choice question per row, reporting each refused row under its cause",
    docs: DOCS,
    warnings: &[],
    causes: &[&ROWS_NONE],
};

pipeline_cause!(PER_OPTION_NONE, "per-option-none", Code::Config, true,
    "checking the synthesis settings",
    "--per-option 0 asks for no state, so no labelled decision could be written",
    "--per-option 0",
    "give --per-option the number of states to write for each option, 1 or more",
    "tests/decide/pipeline.sh::no_state_per_option_is_refused");
pipeline_cause!(RETRIES_NONE, "retries-none", Code::Config, true,
    "checking the synthesis settings",
    "--retry-multiplier 0 allows no attempt, so no state could be written",
    "--retry-multiplier 0",
    "give --retry-multiplier how many attempts each state may take, 1 or more",
    "tests/decide/pipeline.sh::no_attempt_per_state_is_refused");
pipeline_cause!(DOMAIN_BLANK, "schema-domain-blank", Code::Config, false,
    "reading the decision schema {path}",
    "its domain is blank, and the writer is told what to write by it",
    "domain: {domain}",
    "state in the schema's domain what the states are, such as \"a customer support message to an online shoe store\"",
    "tests/decide/pipeline.sh::a_schema_without_a_domain_is_refused");
pipeline_cause!(QUESTIONS_NONE, "schema-questions-none", Code::Config, false,
    "reading the decision schema {path}",
    "it asks no question, so no state would be labelled",
    "questions: 0",
    "add a question to the schema's questions, as docs/guide/decisions.md gives one",
    "tests/decide/pipeline.sh::a_schema_without_questions_is_refused");
pipeline_cause!(NONE_WRITTEN, "none-written", Code::Refused, false,
    "synthesizing labelled decisions about {domain}",
    "the writing model answered no usable state: every answer was empty or repeated an earlier one",
    "attempts: {attempts}; empty: {empty}; repeated: {repeated}; generator: {generator}",
    "raise --temperature or --retry-multiplier, or name another --generator-model",
    "tests/decide/pipeline.sh::a_writer_that_writes_nothing_usable_is_refused");

/// Labelled decisions written by a model.
pub static SYNTHESIZE: Operation = Operation {
    id: "ster.decisions.synthesize",
    entries: &["ster decisions synthesize"],
    uses: &[&REQUEST, &INFERENCE, &EXAMPLES, &FILE],
    summary: "write states for every option of every question a schema asks through Stado's inference alternatives, each labelled with the option it was written for",
    docs: DOCS,
    warnings: &[],
    causes: &[&PER_OPTION_NONE, &RETRIES_NONE, &DOMAIN_BLANK, &QUESTIONS_NONE, &NONE_WRITTEN],
};

pipeline_cause!(HOLDOUT_OUTSIDE, "holdout-outside", Code::Config, true,
    "checking the split settings",
    "--holdout {holdout} is no fraction above zero and below one, so it parts no example from the rest",
    "--holdout {holdout}",
    "give --holdout the share of examples to hold out, such as 0.2",
    "tests/decide/pipeline.sh::a_holdout_outside_zero_and_one_is_refused");

/// A set parted into training and held-out examples.
pub static SPLIT: Operation = Operation {
    id: "ster.decisions.split",
    entries: &["ster decisions split"],
    uses: &[&EXAMPLES, &FILE],
    summary: "part a labelled set into a training set and a held-out set by a seeded shuffle, and write both",
    docs: DOCS,
    warnings: &[],
    causes: &[&HOLDOUT_OUTSIDE],
};

/// A model measured on a labelled set.
pub static BENCHMARK: Operation = Operation {
    id: "ster.decisions.benchmark",
    entries: &["ster decisions benchmark"],
    uses: &[&CALIBRATE, &EXAMPLES, &CALIBRATION, &LOAD, &FILE],
    summary: "answer every labelled question of a set with a model, score accuracy, likelihood and calibration per question type, and write the report",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};
