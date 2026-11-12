//! How creating an untrained checkpoint from a fixture declaration fails
//! (src/fixture): reading the declaration, the Llama config and the
//! tokenizer it names ([`declaration`]), checking what they declare, and
//! writing and verifying the checkpoint ([`output`]).

use std::path::Path;

use wisent_errors::Operation;

use super::{file, refuse};

/// docs/guide/errors/model-fixture.md, where this functionality is listed.
const DOCS: &str = page!("model-fixture");

pub mod declaration;
pub mod output;

/// The refusal of reading `path`, the fixture's `role` (`declaration`,
/// `config`, `tokenizer`), when the system answered `error`.
pub fn read_refusal(role: &str, path: &Path, error: &std::io::Error) -> anyhow::Error {
    let cause = if error.kind() == std::io::ErrorKind::NotFound {
        &declaration::INPUT_MISSING
    } else {
        &declaration::INPUT_UNREADABLE
    };
    refuse(
        cause,
        &[
            ("role", role),
            ("path", &path.display().to_string()),
            ("state", &file::state(path)),
            ("kind", &file::kind(error)),
            ("error", &error.to_string()),
        ],
    )
}

/// The refusal of the fixture's `role` at `path`, which the JSON reader
/// stopped reading with `error`: not JSON, or of another shape.
pub fn json_refusal(role: &str, path: &Path, error: &serde_json::Error) -> anyhow::Error {
    let cause = match error.classify() {
        serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
            &declaration::INPUT_NOT_JSON
        }
        _ => &declaration::INPUT_SHAPE,
    };
    refuse(
        cause,
        &[
            ("role", role),
            ("path", &path.display().to_string()),
            ("line", &error.line().to_string()),
            ("column", &error.column().to_string()),
            ("error", &error.to_string()),
        ],
    )
}

/// Creating a checkpoint from a fixture declaration.
pub static FIXTURE: Operation = Operation {
    id: "ster.model.fixture",
    entries: &["ster model fixture create", "ster request model/fixture/create"],
    uses: &[&super::tokenizer::DECLARATION, &super::model::weights::LOAD, &super::file::FILE],
    summary: "read a fixture declaration, its Llama config and tokenizer, check what they declare, write config.json, tokenizer.json and model.safetensors to a new directory, and verify the checkpoint by loading it and forwarding the declared prompt",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &declaration::INPUT_MISSING,
        &declaration::INPUT_UNREADABLE,
        &declaration::INPUT_NOT_JSON,
        &declaration::INPUT_SHAPE,
        &declaration::MODEL_TYPE,
        &declaration::MLP_BIAS,
        &declaration::HEADS_UNGROUPED,
        &declaration::NOT_FINITE,
        &declaration::SCALE_NEGATIVE,
        &declaration::BIAS_MISSING,
        &declaration::PROMPT_BLANK,
        &declaration::LAYERS_NONE,
        &declaration::LAYER_OUTSIDE,
        &declaration::TOKENIZER_UNPARSED,
        &declaration::TOKENIZER_EMPTY,
        &declaration::TOKEN_OUTSIDE,
        &declaration::TOKENIZER_TOO_WIDE,
        &declaration::WIDTH_OVERFLOW,
        &output::OUTPUT_NOT_UTF8,
        &output::OUTPUT_EXISTS,
        &output::OUTPUT_UNINSPECTABLE,
        &output::OUTPUT_DENIED,
        &output::OUTPUT_UNCREATABLE,
        &output::FILE_UNWRITTEN,
        &output::SAMPLE_NOT_FINITE,
        &output::TENSOR_UNBUILT,
        &output::VERIFICATION_WIDTH,
        &output::VERIFICATION_NOT_FINITE,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&FIXTURE];
