//! How Ster's typed decisions fail: reading and checking a decision request,
//! reading and writing a calibration, reading labelled examples, answering a
//! request (`ster decide`) and fitting a temperature (`ster calibrate`).
//! Reading and writing their files is the `ster.file` layer's.

use wisent_errors::{Cause, Code, Operation};

use super::file::FILE;
use super::model::weights::LOAD;

pub mod calibration;
#[path = "remote/fetch.rs"]
pub mod fetch;
#[path = "remote/pipeline.rs"]
pub mod pipeline;
pub mod request;
pub mod rows;

/// docs/guide/errors/decide.md, where reading a request and answering it are
/// listed.
const DOCS: &str = page!("decide");

/// docs/guide/errors/decide-calibration.md, where the calibration and example
/// layers and fitting a temperature are listed.
const CALIBRATION_DOCS: &str = page!("decide-calibration");

pub static QUESTION_TOO_LONG: Cause = Cause {
    id: "question-too-long",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "rendering question '{id}'",
    cause: "its prompt is {tokens} tokens, and this model reads at most {context} at once",
    observed: "tokens: {tokens}; context: {context}",
    remedy: "shorten the state or the question's options, or decide with a model whose context holds the prompt",
    test: "tests/decide/remote/request.sh::a_question_longer_than_the_context_is_refused",
};

pub static LETTER_OUTSIDE: Cause = Cause {
    id: "letter-outside",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the answer under letter {letter}",
    cause: "the tokenizer spells {letter} as token {token}, and the model scores only tokens 0 to {vocabulary} exclusive, so its tokenizer and weights are of different vocabularies",
    observed: "token: {token}; scored tokens: {vocabulary}",
    remedy: "load the checkpoint with the tokenizer it was trained with; its config's vocab_size counts every token the tokenizer can produce",
    test: "tests/decide/remote/request.sh::a_letter_the_model_does_not_score_is_refused",
};

pub static BINS_NONE: Cause = Cause {
    id: "bins-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the calibration settings",
    cause: "the expected calibration error is measured over equal-width confidence bins, and zero bins measure nothing",
    observed: "--ece-bins {bins}",
    remedy: "give --ece-bins a whole number of 1 or more, such as 10",
    test: "tests/decide/calibration.sh::zero_confidence_bins_are_refused",
};

/// Reading and checking one decision document.
pub static REQUEST: Operation = Operation {
    id: "ster.decide.request",
    entries: &[],
    uses: &[&FILE],
    summary: "read a decision request from a file or standard input and check its state and every question before a weight is mapped",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &request::MISSING,
        &request::STATE_EMPTY,
        &request::QUESTIONS_NONE,
        &request::ID_EMPTY,
        &request::INSTRUCTIONS_EMPTY,
        &request::OPTIONS_TOO_FEW,
        &request::OPTIONS_TOO_MANY,
        &request::OPTION_NAME_EMPTY,
        &request::LEVEL_EMPTY,
    ],
};

/// Reading, checking and writing one calibration artifact.
pub static CALIBRATION: Operation = Operation {
    id: "ster.decide.calibration",
    entries: &[],
    uses: &[&FILE],
    summary: "read a calibration artifact, check its schema, temperature and model, and write one back",
    docs: CALIBRATION_DOCS,
    warnings: &[],
    causes: &[
        &calibration::CALIBRATION_MISSING,
        &calibration::CALIBRATION_SCHEMA,
        &calibration::CALIBRATION_TEMPERATURE,
        &calibration::CALIBRATION_MODEL,
    ],
};

/// Reading and checking one labelled example set.
pub static EXAMPLES: Operation = Operation {
    id: "ster.decide.examples",
    entries: &[],
    uses: &[&REQUEST, &FILE],
    summary: "read a labelled example set and check every request in it and every label against the options it names",
    docs: CALIBRATION_DOCS,
    warnings: &[],
    causes: &[
        &calibration::EXAMPLES_MISSING,
        &calibration::EXAMPLES_NONE,
        &calibration::LABEL_UNASKED,
        &calibration::LABEL_NOT_OPTION,
        &calibration::LABELS_NONE,
    ],
};

/// Answering every question of one request.
pub static DECIDE: Operation = Operation {
    id: "ster.decide",
    entries: &["ster decide", "ster request decide"],
    uses: &[&REQUEST, &CALIBRATION, &LOAD, &FILE],
    summary: "answer every typed question of a decision request with the probability the model puts on each option",
    docs: DOCS,
    warnings: &[],
    causes: &[&QUESTION_TOO_LONG, &LETTER_OUTSIDE],
};

/// Fitting a temperature on labelled examples.
pub static CALIBRATE: Operation = Operation {
    id: "ster.decide.calibrate",
    entries: &["ster calibrate", "ster request calibrate"],
    uses: &[&EXAMPLES, &CALIBRATION, &LOAD, &DECIDE],
    summary: "fit the temperature that makes a model's decision probabilities match labelled examples, and write it as a calibration",
    docs: CALIBRATION_DOCS,
    warnings: &[],
    causes: &[&BINS_NONE],
};

pub(super) static OPERATIONS: &[&Operation] = &[
    &REQUEST,
    &CALIBRATION,
    &EXAMPLES,
    &DECIDE,
    &CALIBRATE,
    &pipeline::QUESTION,
    &fetch::FETCH,
    &rows::ROWS,
    &pipeline::IMPORT,
    &pipeline::SYNTHESIZE,
    &pipeline::SPLIT,
    &pipeline::BENCHMARK,
];
