//! How reading a prompt set fails (`workflow::PromptSet`, the
//! `{"prompts": ["…"]}` document `extract`, `generate --prompts`, `converse`
//! and `tune grpo` read), and `ster extract`, which writes the states a
//! model reaches for each prompt. A file that cannot be read or written, or
//! is not JSON of the set's form, is the `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::super::chat::TEMPLATE;
use super::super::file::FILE;
use super::super::model::weights::LOAD;
use super::super::runtime::encode::ENCODE;
use super::DOCS;
use super::layers::LAYERS;

/// What a prompt set is, for the `ster.file` refusals that name it.
pub const FORM: &str = "a JSON object {\"prompts\": [\"…\"]} listing one prompt per string";

pub static MISSING: Cause = Cause {
    id: "prompts-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading the prompt set {path}",
    cause: "no file is there: nothing at {path} is the prompt set named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name a prompt set, a JSON object whose \"prompts\" lists one prompt per string as docs/examples/prompts.json does; `ls -ln {parent}` shows what that directory holds",
    test: "tests/steering/workflow.sh::a_missing_prompt_set_is_refused",
};

pub static NONE: Cause = Cause {
    id: "prompts-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the prompt set {path}",
    cause: "prompt set contains no prompts, so there is nothing to read the model with",
    observed: "path: {path} ({state}); prompts: []",
    remedy: "list at least one prompt under \"prompts\" in {path}",
    test: "tests/steering/workflow.sh::an_empty_prompt_set_is_refused",
};

/// Reading a prompt set.
pub static PROMPTS: Operation = Operation {
    id: "ster.prompts",
    entries: &[],
    uses: &[&FILE],
    summary: "read a JSON object whose \"prompts\" lists at least one prompt",
    docs: DOCS,
    warnings: &[],
    causes: &[&MISSING, &NONE],
};

pub static STATES_NOT_FINITE: Cause = Cause {
    id: "states-not-finite",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading prompt {index} of {prompts} at layer {layer}",
    cause: "the state the model reached holds a value that is not finite, and JSON has no spelling for one, so it would be written as null: the forward pass overflowed at this precision, or the weights hold one",
    observed: "prompt {index} at layer {layer}: value {position} of {width} is {value}; precision {precision}; nothing was written to {output}",
    remedy: "read the prompts at a wider precision (--precision f32), or check the checkpoint's weights are finite",
    test: "tests/steering/workflow.sh::states_that_are_not_finite_are_refused",
};

/// `ster extract`: the hidden states a model reaches for each prompt.
pub static EXTRACT: Operation = Operation {
    id: "ster.extract",
    entries: &["ster extract", "ster request extract"],
    uses: &[&PROMPTS, &LOAD, &TEMPLATE, &LAYERS, &ENCODE, &FILE],
    summary: "read the prompt set before any weight, read each prompt through the model at the layers named, and write every layer's last-position state as one JSON document",
    docs: DOCS,
    warnings: &[],
    causes: &[&STATES_NOT_FINITE],
};
