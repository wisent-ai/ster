//! Why one row of a benchmark export cannot become a pair
//! (src/pairs/benchmark/{published,livecodebench,choices}.rs). `ster pairs
//! import` lists every skipped row in its report under its cause, and
//! refuses an export no row of which becomes a pair with the first skipped
//! row's cause, so its fix is the row's own.

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/pairs-import.md, where this layer is listed.
const DOCS: &str = page!("pairs-import");

macro_rules! row_cause {
    ($name:ident, $id:literal, $cause:expr, $observed:expr, $remedy:expr, $test:literal) => {
        pub static $name: Cause = Cause {
            id: $id,
            code: Code::Config,
            usage: false,
            exit: None,
            failed: "importing {row} of {source}",
            cause: $cause,
            observed: $observed,
            remedy: $remedy,
            test: concat!("tests/pairs/imports/import-rows.sh::", $test),
        };
    };
}

// The CSV exports: TruthfulQA, Do-Not-Answer, BiPO.
row_cause!(QUESTION_BLANK, "question-blank",
    "the {column} column of {row} is blank, and every pair asks a question",
    "{cells}",
    "write the question of {row} in {source}, or remove the row",
    "a_row_without_a_question_is_skipped");
row_cause!(ANSWER_BLANK, "answer-blank",
    "the {column} column of {row} is blank, and it is the {side} side of the pair",
    "{cells}",
    "write the {column} of {row} in {source}, or remove the row",
    "a_row_with_a_blank_answer_is_skipped");
row_cause!(INCORRECT_NONE, "incorrect-none",
    "the Incorrect Answers column of {row} holds no answer once split at `;`, and the negative side is one of them",
    "{cells}",
    "write at least one incorrect answer of {row}, answers parted by `;`, or remove the row",
    "a_row_without_an_incorrect_answer_is_skipped");

// LiveCodeBench: problems.json against question_examples.json.
row_cause!(PROBLEM_ID_MISSING, "problem-id-missing",
    "{row} has no question_id text, which is what pairs a problem with its examples",
    "question_id: {found}",
    "give {row} of {source} the question_id its examples carry",
    "a_problem_without_an_id_is_skipped");
row_cause!(PROBLEM_CONTENT_BLANK, "problem-content-blank",
    "{row} has no question_content text, and it is the question of the pair",
    "question_content: {found}",
    "give {row} of {source} its question_content",
    "a_problem_without_content_is_skipped");
row_cause!(EXAMPLE_MISSING, "example-missing",
    "no question of {examples} carries the question_id of {row}, {id}, and the pair's code comes from its examples",
    "questions in {examples}: {count}",
    "name the question_examples.json written with {source} through --examples, or add the examples of {id} to it",
    "a_problem_without_examples_is_skipped");
row_cause!(EXAMPLE_CODE_MISSING, "example-code-missing",
    "the example of {id} in {examples} has no {missing} code, and the pair is its good code against its bad",
    "good_example.code: {good}; bad_example.code: {bad}",
    "write the {missing} code of {id} in {examples}, or remove the example",
    "an_example_without_code_is_skipped");

// --benchmark choices: the parts a row holds where the row flags point.
row_cause!(CHOICE_QUESTION_MISSING, "choice-question-missing",
    "{row} has no question text at {pointer}, where --question points",
    "{pointer}: {found}",
    "point --question at the question of every row, or remove {row} from {source}",
    "a_choice_row_without_a_question_is_skipped");
row_cause!(CHOICES_NOT_LIST, "choices-not-list",
    "{row} holds no list at {pointer}, where --choices points",
    "{pointer}: {found}",
    "point --choices at the list of choices of every row",
    "choices_that_are_not_a_list_are_skipped");
row_cause!(CHOICE_NOT_TEXT, "choice-not-text",
    "choice {index} of {row} is {found}, where every choice is non-empty text",
    "{pointer}: {choices}",
    "write choice {index} of {row} in {source}, or remove the row",
    "a_choice_that_is_not_text_is_skipped");
row_cause!(CHOICES_FEW, "choices-few",
    "{row} holds {count} choices at {pointer}, and a pair is the correct choice against another",
    "{pointer}: {choices}",
    "give {row} at least two choices, or remove it from {source}",
    "a_row_with_one_choice_is_skipped");
row_cause!(ANSWER_MISSING, "answer-missing",
    "{row} has no answer at {pointer}, where --answer points",
    "{pointer}: {found}",
    "point --answer at the answer of every row, or remove {row} from {source}",
    "a_row_without_an_answer_is_skipped");
row_cause!(ANSWER_NOT_INDEX, "answer-not-index",
    "the answer of {row} at {pointer} is {found}, where --answer-form index reads a whole number from 0, or text holding one",
    "{detail}",
    "give --answer-form the form the export's answers take (index, label or text), or correct the answer of {row}",
    "an_answer_that_is_not_an_index_is_skipped");
row_cause!(ANSWER_INDEX_OUTSIDE, "answer-index-outside",
    "the answer of {row} names choice {index}, and the row holds {count} choices, numbered from 0 to {last}",
    "choices: {choices}",
    "correct the answer of {row}, or point --choices at the list its answer counts in",
    "an_answer_past_the_choices_is_skipped");
row_cause!(LABELS_NOT_LIST, "labels-not-list",
    "{row} holds no list at {pointer}, where --labels points",
    "{pointer}: {found}",
    "point --labels at the list of labels of every row",
    "labels_that_are_not_a_list_are_skipped");
row_cause!(LABEL_NOT_TEXT, "label-not-text",
    "label {index} of {row} is {found}, where every label is non-empty text",
    "{pointer}: {labels}",
    "write label {index} of {row} in {source}, or remove the row",
    "a_label_that_is_not_text_is_skipped");
row_cause!(LABELS_MISALIGNED, "labels-misaligned",
    "{row} holds {labels} labels and {choices} choices, and --answer-form label reads the label at each choice's position",
    "labels: {shown_labels}; choices: {shown_choices}",
    "give {row} one label per choice, or point --labels at the list that lines up with --choices",
    "labels_that_do_not_line_up_are_skipped");
row_cause!(ANSWER_LABEL_NOT_TEXT, "answer-label-not-text",
    "the answer of {row} at {pointer} is {found}, where --answer-form label reads one of the row's labels as text",
    "labels: {labels}",
    "give --answer-form the form the export's answers take (index, label or text), or correct the answer of {row}",
    "an_answer_that_is_not_a_label_is_skipped");
row_cause!(ANSWER_LABEL_UNKNOWN, "answer-label-unknown",
    "the answer of {row}, {answer}, is none of its labels",
    "labels: {labels}",
    "correct the answer of {row} to one of its labels, or point --labels at the list its answer names",
    "an_answer_that_names_no_label_is_skipped");
row_cause!(ANSWER_TEXT_NOT_TEXT, "answer-text-not-text",
    "the answer of {row} at {pointer} is {found}, where --answer-form text reads the correct choice's text",
    "choices: {choices}",
    "give --answer-form the form the export's answers take (index, label or text), or correct the answer of {row}",
    "an_answer_that_is_not_text_is_skipped");
row_cause!(ANSWER_TEXT_UNKNOWN, "answer-text-unknown",
    "the answer of {row}, {answer}, is none of its choices",
    "choices: {choices}",
    "correct the answer of {row} to the text of one of its choices",
    "an_answer_that_names_no_choice_is_skipped");

/// Why one row of a benchmark export cannot become a pair.
pub static ROWS: Operation = Operation {
    id: "ster.pairs.import.rows",
    entries: &[],
    uses: &[],
    summary: "read one row of a TruthfulQA, Do-Not-Answer, BiPO, LiveCodeBench or multiple-choice export into one pair, or name why it cannot be one",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &QUESTION_BLANK,
        &ANSWER_BLANK,
        &INCORRECT_NONE,
        &PROBLEM_ID_MISSING,
        &PROBLEM_CONTENT_BLANK,
        &EXAMPLE_MISSING,
        &EXAMPLE_CODE_MISSING,
        &CHOICE_QUESTION_MISSING,
        &CHOICES_NOT_LIST,
        &CHOICE_NOT_TEXT,
        &CHOICES_FEW,
        &ANSWER_MISSING,
        &ANSWER_NOT_INDEX,
        &ANSWER_INDEX_OUTSIDE,
        &LABELS_NOT_LIST,
        &LABEL_NOT_TEXT,
        &LABELS_MISALIGNED,
        &ANSWER_LABEL_NOT_TEXT,
        &ANSWER_LABEL_UNKNOWN,
        &ANSWER_TEXT_NOT_TEXT,
        &ANSWER_TEXT_UNKNOWN,
    ],
};
