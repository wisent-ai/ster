//! How comparing a model's forward pass with a reference implementation's
//! recorded states fails (src/workflow/parity): the parity file, each
//! record's token ids, the reference safetensors file and the states it
//! holds. A file that cannot be read, or is not JSON or safetensors of the
//! form named, is the `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::super::file::FILE;
use super::super::model::weights::LOAD;
use super::DOCS;

/// What a parity file is, for the `ster.file` refusals that name it.
pub const FORM: &str = "a JSON object whose \"records\" each name \"hidden\" (a safetensors file) and \"tokenIds\" or \"tokenIdsFile\"";

/// What a token id file is, for the `ster.file` refusals that name it.
pub const IDS_FORM: &str = "a JSON list of whole token ids, such as [1, 2, 3]";

const RECORD_STEP: &str = "comparing record {index} of the parity file {input}";

pub static INPUT_MISSING: Cause = Cause {
    id: "input-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading the parity file {path}",
    cause: "no file is there: nothing at {path} is the parity file --input names",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name the parity file that lists the records to compare; `ls -ln {parent}` shows what that directory holds",
    test: "tests/steering/parity.sh::a_missing_parity_file_is_refused",
};

pub static RECORDS_NONE: Cause = Cause {
    id: "records-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the parity file {input}",
    cause: "{input} lists no records to compare, so there is no state to measure",
    observed: "path: {input} ({state}); records: []",
    remedy: "list at least one record under \"records\" in {input}, each naming the token ids (tokenIds) and the states file (hidden)",
    test: "tests/steering/parity.sh::a_parity_file_without_records_is_refused",
};

pub static IDS_BOTH: Cause = Cause {
    id: "ids-both",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "record {index} names both tokenIds and tokenIdsFile; give one, so it is clear which ids the reference read",
    observed: "record {index}: tokenIds holds {count} ids; tokenIdsFile: {file}",
    remedy: "keep tokenIds or tokenIdsFile in record {index}, not both",
    test: "tests/steering/parity.sh::a_record_naming_two_id_sources_is_refused",
};

pub static IDS_NEITHER: Cause = Cause {
    id: "ids-neither",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "record {index} names neither tokenIds nor tokenIdsFile, so there are no tokens to read",
    observed: "record {index}: hidden {hidden}; no tokenIds, no tokenIdsFile",
    remedy: "give record {index} the token ids the reference read, as tokenIds or as a file named by tokenIdsFile",
    test: "tests/steering/parity.sh::a_record_naming_no_ids_is_refused",
};

pub static IDS_FILE_MISSING: Cause = Cause {
    id: "ids-file-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "no file is there: nothing at {path} is the token id file the record names; tokenIdsFile is read relative to the parity file's directory",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name the token id file relative to the parity file's directory; `ls -ln {parent}` shows what it holds",
    test: "tests/steering/parity.sh::a_missing_token_id_file_is_refused",
};

pub static IDS_NONE: Cause = Cause {
    id: "ids-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "record {index} holds no token ids, so there is no position to compare",
    observed: "record {index}: {source} holds []",
    remedy: "give record {index} the token ids the reference read",
    test: "tests/steering/parity.sh::a_record_without_ids_is_refused",
};

pub static TOKEN_PAST: Cause = Cause {
    id: "token-past",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "record {index} holds token id {id}, past the model's {vocabulary} tokens: the reference read another tokenizer's ids",
    observed: "record {index}: id {id} at position {position}; the model's embedding table holds ids 0 to {last}",
    remedy: "record the reference states with the tokenizer of the model --model names, or compare against the model the ids belong to",
    test: "tests/steering/parity.sh::a_token_past_the_vocabulary_is_refused",
};

pub static STATES_MISSING: Cause = Cause {
    id: "states-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "no file is there: nothing at {path} is the reference states file the record's hidden names; hidden is read relative to the parity file's directory",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name the safetensors file the reference wrote, relative to the parity file's directory; `ls -ln {parent}` shows what it holds",
    test: "tests/steering/parity.sh::a_missing_states_file_is_refused",
};

pub static TENSOR_ABSENT: Cause = Cause {
    id: "tensor-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "{path} holds no tensor \"{tensor}\"; the record names it, or names none and hidden_states is read",
    observed: "{path}: tensors {names}",
    remedy: "name the tensor that holds the states with the record's tensor field, one of {names}",
    test: "tests/steering/parity.sh::a_tensor_the_file_lacks_is_refused",
};

pub static TENSOR_SHAPE: Cause = Cause {
    id: "tensor-shape",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "{path}'s \"{tensor}\" is {shape}; Ster compares [positions, hidden] states, or [1, positions, hidden] of one sequence",
    observed: "{path}: \"{tensor}\" of shape {shape}, {dtype}",
    remedy: "save the residual stream of one sequence after the final norm as [positions, hidden]",
    test: "tests/steering/parity.sh::a_tensor_of_another_shape_is_refused",
};

pub static POSITIONS_PAST: Cause = Cause {
    id: "positions-past",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "record {index}'s reference holds {positions} positions for {tokens} tokens: the reference read more tokens than the record lists",
    observed: "{path}: [{positions}, {width}]; record {index}: {tokens} token ids",
    remedy: "list every token id the reference read in record {index}, or save only the positions of those ids",
    test: "tests/steering/parity.sh::more_positions_than_tokens_are_refused",
};

pub static WIDTH_OTHER: Cause = Cause {
    id: "width-other",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "record {index}'s reference states are {width} wide and this model's {ours}: they come from a model of another width",
    observed: "{path}: [{positions}, {width}]; {model}: hidden_size {ours}",
    remedy: "compare against states the reference computed for {model}, or name the model they came from with --model",
    test: "tests/steering/parity.sh::states_of_another_width_are_refused",
};

pub static POSITIONS_NONE: Cause = Cause {
    id: "positions-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: RECORD_STEP,
    cause: "record {index}'s \"{tensor}\" in {path} holds no positions to compare",
    observed: "{path}: \"{tensor}\" of shape {shape}",
    remedy: "save the states of at least one position",
    test: "tests/steering/parity.sh::a_tensor_without_positions_is_refused",
};

/// `ster parity`: how far a model's states lie from a reference's.
pub static PARITY: Operation = Operation {
    id: "ster.parity",
    entries: &["ster parity", "ster request parity"],
    uses: &[&FILE, &LOAD],
    summary: "read every record's token ids and reference states, run the model over the same ids and measure how far its final states lie from the reference's",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &INPUT_MISSING,
        &RECORDS_NONE,
        &IDS_BOTH,
        &IDS_NEITHER,
        &IDS_FILE_MISSING,
        &IDS_NONE,
        &TOKEN_PAST,
        &STATES_MISSING,
        &TENSOR_ABSENT,
        &TENSOR_SHAPE,
        &POSITIONS_PAST,
        &WIDTH_OTHER,
        &POSITIONS_NONE,
    ],
};
