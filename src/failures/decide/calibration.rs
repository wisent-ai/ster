//! How reading and writing a calibration artifact fails
//! (src/decide/calibration.rs), and how reading and checking the labelled
//! examples a calibration is fitted on fails (`ExampleSet` in
//! src/decide/examples.rs). A file that cannot be read or written, or is not
//! JSON of its form, is the `ster.file` layer's cause.

use wisent_errors::{Cause, Code};

/// What a calibration is, for the `ster.file` refusals that name it.
pub const CALIBRATION_FORM: &str = "the JSON `ster calibrate --output <file>` writes";

/// What a labelled example set is, for the `ster.file` refusals that name it.
pub const EXAMPLES_FORM: &str = "{\"examples\": [{\"request\": <decision request>, \"answers\": {\"<question id>\": <the option it names>}}]}, as `ster decisions import` and `ster decisions synthesize` write it";

pub static CALIBRATION_MISSING: Cause = Cause {
    id: "calibration-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading calibration {path}",
    cause: "no file is there: nothing at {path} is the calibration named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name a calibration `ster calibrate --output {path}` wrote for this model (`ls -ln {parent}` shows what that directory holds), or leave --calibration out to read raw probabilities",
    test: "tests/decide/calibration.sh::a_missing_calibration_is_refused",
};

pub static CALIBRATION_SCHEMA: Cause = Cause {
    id: "calibration-schema",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading calibration {path}",
    cause: "it declares schema \"{schema}\", and this Ster reads {expected}",
    observed: "schema: \"{schema}\"",
    remedy: "fit it again with this Ster: `ster calibrate --examples <labelled.json> --output {path}`",
    test: "tests/decide/calibration.sh::a_calibration_of_another_schema_is_refused",
};

pub static CALIBRATION_TEMPERATURE: Cause = Cause {
    id: "calibration-temperature",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading calibration {path}",
    cause: "its temperature is {temperature}, and the logits are divided by it, so only a positive number is a temperature",
    observed: "temperature: {temperature}",
    remedy: "fit it again with `ster calibrate --examples <labelled.json> --output {path}`; a fitted temperature is always positive",
    test: "tests/decide/calibration.sh::a_calibration_with_a_temperature_of_zero_is_refused",
};

pub static CALIBRATION_MODEL: Cause = Cause {
    id: "calibration-model",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "applying calibration {path}",
    cause: "it was fitted for model {fitted}, and a temperature fitted on one checkpoint says nothing about another",
    observed: "calibration model: {fitted}; loaded model: {model}",
    remedy: "fit a calibration for {model} with `ster calibrate --model {model} --examples <labelled.json> --output <file>`, or name the calibration fitted for it",
    test: "tests/decide/calibration.sh::a_calibration_for_another_model_is_refused",
};

pub static EXAMPLES_MISSING: Cause = Cause {
    id: "examples-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading labelled examples {path}",
    cause: "no file is there: nothing at {path} is the labelled example set named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name an existing labelled example set with --examples (`ls -ln {parent}` shows what that directory holds), or write one with `ster decisions import` or `ster decisions synthesize`",
    test: "tests/decide/examples.sh::a_missing_example_set_is_refused",
};


pub static EXAMPLES_NONE: Cause = Cause {
    id: "examples-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the labelled examples",
    cause: "the set holds no example, and a temperature is fitted on labelled examples",
    observed: "{source}: \"examples\": []",
    remedy: "add at least one example with a request and the answer a person gave to one of its questions",
    test: "tests/decide/examples.sh::an_empty_example_set_is_refused",
};

pub static LABEL_UNASKED: Cause = Cause {
    id: "label-unasked",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking labelled example {index}",
    cause: "it labels question '{id}', which its request does not ask",
    observed: "questions asked: {asked}",
    remedy: "label one of the questions its request asks, or add question '{id}' to the request",
    test: "tests/decide/examples.sh::a_label_for_a_question_not_asked_is_refused",
};

pub static LABEL_NOT_OPTION: Cause = Cause {
    id: "label-not-option",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking labelled example {index}",
    cause: "it answers question '{id}' with {answer}, which is not one of its options",
    observed: "answer: {answer}; options: {options}",
    remedy: "answer a choice question with one of its option names, a score question with a level index from 0, and a noul question with true or false",
    test: "tests/decide/examples.sh::a_label_that_is_not_an_option_is_refused",
};

pub static LABELS_NONE: Cause = Cause {
    id: "labels-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the labelled examples",
    cause: "none of its {count} examples labels a question, so there is nothing to fit on",
    observed: "examples: {count}; labelled questions: 0",
    remedy: "give at least one example an answers entry naming the option a person chose",
    test: "tests/decide/examples.sh::an_example_set_without_labels_is_refused",
};
