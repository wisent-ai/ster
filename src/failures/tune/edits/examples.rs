//! How reading an example set fails (src/tune/examples.rs): the file a
//! supervised run and `ster tune evaluate` read prompts and completions from.
//! A file that cannot be read, or is not JSON of an example set's form, is
//! the `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;
use crate::failures::file::FILE;

/// What an example set is, for the `ster.file` refusals that name it.
pub const FORM: &str = "{\"examples\": [{\"prompt\": \"<text>\", \"completion\": \"<text>\"}]}, as docs/examples/tuning/examples.json is";

pub static MISSING: Cause = Cause {
    id: "example-set-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading example set {path}",
    cause: "no file is there: nothing at {path} is the example set named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name an existing example-set file, shaped as docs/examples/tuning/examples.json is; `ls -ln {parent}` shows what that directory holds",
    test: "tests/tuning/runs/examples.sh::a_missing_example_set_is_refused",
};

pub static EMPTY: Cause = Cause {
    id: "example-set-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading example set {label}",
    cause: "example set {label} contains no examples",
    observed: "examples: 0; name: \"{name}\"",
    remedy: "add at least one {\"prompt\": \"<text>\", \"completion\": \"<text>\"} to its examples",
    test: "tests/tuning/runs/examples.sh::an_example_set_without_examples_is_refused",
};

pub static BLANK: Cause = Cause {
    id: "example-blank",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading example set {label}",
    cause: "example set {label} contains an empty prompt or completion",
    observed: "example {index}: prompt {prompt}, completion {completion}",
    remedy: "give example {index} text in both its prompt and its completion, or remove it",
    test: "tests/tuning/runs/examples.sh::an_example_with_an_empty_side_is_refused",
};

pub static SYSTEM_BLANK: Cause = Cause {
    id: "system-blank",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading example set {label}",
    cause: "example set {label} contains an empty system prompt",
    observed: "example {index}: system {system}",
    remedy: "give example {index} the text of its system turn, or remove its \"system\" key",
    test: "tests/tuning/runs/examples.sh::an_empty_system_prompt_is_refused",
};

/// Reading and checking one example-set file.
pub static EXAMPLES: Operation = Operation {
    id: "ster.tune.examples",
    entries: &[],
    uses: &[&FILE],
    summary: "read an example-set file of prompts and completions and check every example carries text",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &MISSING,
        &EMPTY,
        &BLANK,
        &SYSTEM_BLANK,
    ],
};
