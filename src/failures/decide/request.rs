//! How reading and checking a decision document fails (src/decide/request.rs
//! and the `--request` reading of src/cli/decide): the file or standard input
//! read, the JSON decoded, and every question checked before a weight is
//! mapped. A file that cannot be read, or is not JSON of a request's form,
//! is the `ster.file` layer's cause.

use wisent_errors::{Cause, Code};

/// What a decision request is, for the `ster.file` refusals that name it.
pub const FORM: &str = "{\"state\": <text or JSON>, \"questions\": {\"<id>\": {\"type\": \"choice\", \"instructions\": \"<text>\", \"criteria\": {\"<option>\": null, \"<option>\": null}}}}, a question's type being choice, score or noul, as docs/guide/decisions.md gives it";

pub static MISSING: Cause = Cause {
    id: "request-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading decision request {path}",
    cause: "no file is there: nothing at {path} is the decision request named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name an existing decision document with --request (`ls -ln {parent}` shows what that directory holds), or give `--request -` and write it on standard input",
    test: "tests/decide/remote/request.sh::a_missing_request_is_refused",
};

pub static STATE_EMPTY: Cause = Cause {
    id: "state-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the decision request",
    cause: "its state is empty, so every question would be judged against nothing",
    observed: "state: {state}",
    remedy: "give the request a non-empty state: the text or JSON the questions are about",
    test: "tests/decide/remote/request.sh::an_empty_state_is_refused",
};

pub static QUESTIONS_NONE: Cause = Cause {
    id: "questions-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the decision request",
    cause: "it asks no question",
    observed: "questions: {questions}",
    remedy: "add a question under questions: an object keyed by the question's id, holding its `type` (choice, score or noul) and its `instructions`",
    test: "tests/decide/remote/request.sh::a_request_without_questions_is_refused",
};

pub static ID_EMPTY: Cause = Cause {
    id: "question-id-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the decision request",
    cause: "a question's id is empty, and the answer comes back under that id",
    observed: "question id: \"{id}\"",
    remedy: "give the question an id with at least one character other than white space",
    test: "tests/decide/remote/request.sh::a_question_with_an_empty_id_is_refused",
};

pub static INSTRUCTIONS_EMPTY: Cause = Cause {
    id: "instructions-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking question '{id}'",
    cause: "its instructions are empty, so the model is shown options with no question",
    observed: "instructions: {instructions}",
    remedy: "write the question in its instructions field",
    test: "tests/decide/remote/request.sh::a_question_without_instructions_is_refused",
};

pub static OPTIONS_TOO_FEW: Cause = Cause {
    id: "options-too-few",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking {kind} question '{id}'",
    cause: "it offers {count} {unit}, and a decision needs at least two to choose between",
    observed: "{unit}: {count}",
    remedy: "give its criteria at least two {unit}; a yes or no question is a noul question",
    test: "tests/decide/remote/request.sh::a_question_with_one_option_is_refused",
};

pub static OPTIONS_TOO_MANY: Cause = Cause {
    id: "options-too-many",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking {kind} question '{id}'",
    cause: "it offers {count} {unit}, and each is answered under one of the {max} letters A to Z read at one token position",
    observed: "{unit}: {count}",
    remedy: "split the question into questions of at most {max} {unit} each",
    test: "tests/decide/remote/request.sh::a_question_with_more_options_than_letters_is_refused",
};

pub static OPTION_NAME_EMPTY: Cause = Cause {
    id: "option-name-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking choice question '{id}'",
    cause: "one of its options has an empty name, and the answer names the option chosen",
    observed: "option name: \"{name}\"",
    remedy: "give every key of its criteria at least one character other than white space",
    test: "tests/decide/remote/request.sh::an_option_with_an_empty_name_is_refused",
};

pub static LEVEL_EMPTY: Cause = Cause {
    id: "level-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking score question '{id}'",
    cause: "its level {index} has an empty description, so the model is shown a letter with nothing under it",
    observed: "level {index}: {level}",
    remedy: "describe every level of its criteria, in order from the first",
    test: "tests/decide/remote/request.sh::a_level_without_a_description_is_refused",
};
