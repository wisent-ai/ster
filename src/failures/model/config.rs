//! How reading a checkpoint's config into a decoder fails: the file, the
//! family it names, the keys Candle's Llama config needs, and every setting a
//! family's reading refuses (src/runtime/load/config.rs, checkpoint/mod.rs,
//! family/, mistral/).

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/model-config.md, where this functionality is listed.
const DOCS: &str = page!("model-config");

/// What a model config is, for the `ster.file` refusals that name it.
pub const FORM: &str = "the config.json the checkpoint's publisher wrote, one JSON object as docs/examples/model/config.json is";

pub static NOT_OBJECT: Cause = Cause {
    id: "config-not-object",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading model config {path}",
    cause: "it is JSON, and not one object of settings: a config states each setting under its key",
    observed: "path: {path}; it holds {held}",
    remedy: "use the checkpoint's config as its publisher wrote it, one JSON object, as docs/examples/model/config.json is",
    test: "tests/models/loading/load.sh::a_config_that_is_not_json_is_refused",
};

pub static MODEL_TYPE_ABSENT: Cause = Cause {
    id: "model-type-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the model type of {path}",
    cause: "it states no model_type, and its architectures ({architectures}) name no model class Ster reads in its place",
    observed: "model_type: absent; architectures: {architectures}",
    remedy: "use a Hugging Face checkpoint whose model_type is one of {families}",
    test: "tests/models/loading/load.sh::a_config_naming_no_model_type_is_refused",
};

pub static MODEL_TYPE_NOT_TEXT: Cause = Cause {
    id: "model-type-not-text",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the model type of {path}",
    cause: "it states model_type as {value}, which is not a string",
    observed: "model_type: {value}",
    remedy: "use a Hugging Face checkpoint whose model_type is one of {families}",
    test: "tests/models/loading/load.sh::a_model_type_that_is_not_text_is_refused",
};

pub static FAMILY_UNKNOWN: Cause = Cause {
    id: "family-unknown",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "choosing the decoder for {path}",
    cause: "model architecture \"{model_type}\" is unsupported by this Ster build",
    observed: "model_type: \"{model_type}\"",
    remedy: "use a Hugging Face checkpoint whose model_type is one of {families}",
    test: "tests/models/loading/load.sh::an_unknown_model_type_is_refused",
};

pub static QUANTIZED: Cause = Cause {
    id: "quantized",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "choosing the decoder for {path}",
    cause: "{path} is a quantized checkpoint (it declares quantization_config); Ster maps unquantized safetensors only",
    observed: "quantization_config: {value}",
    remedy: "use the checkpoint it was quantized from",
    test: "tests/models/loading/load.sh::a_quantized_checkpoint_is_refused",
};

pub static KEY_ABSENT: Cause = Cause {
    id: "key-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the {model_type} decoder's {key} from {path}",
    cause: "{path} states no {key}{spellings}; Ster needs the {model_type} decoder's {key} ({kind})",
    observed: "{key}: absent",
    remedy: "add {key} to the config as {kind}",
    test: "tests/models/loading/load.sh::a_config_without_a_required_key_is_refused",
};

pub static KEY_MISTYPED: Cause = Cause {
    id: "key-mistyped",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {key} from {path}",
    cause: "{path} states {key} as {value}, which is not {kind}",
    observed: "{key}: {value}",
    remedy: "state {key} as {kind}, the type the publisher's config class reads it as",
    test: "tests/models/loading/load.sh::a_key_of_the_wrong_kind_is_refused",
};

pub static UNDECODABLE: Cause = Cause {
    id: "config-undecodable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the {model_type} decoder's settings from {path}",
    cause: "Candle's Llama config reader needs a key this config leaves out or states with another type than it reads, after every key Ster checks itself passed",
    observed: "path: {path}; the reader answered: {error}",
    remedy: "state the key the reader names with the type it names, as the publisher's config does",
    test: "tests/models/loading/load.sh::a_config_candle_cannot_read_is_refused",
};

/// What every settings refusal's `{detail}`, the sentence its site composes,
/// sits in: the config, the key and what the decoder implements.
const SETTINGS_OBSERVED: &str = "{detail}; the config states {observed}";

pub static SETTINGS_UNIMPLEMENTED: Cause = Cause {
    id: "settings-unimplemented",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the decoder's settings from the checkpoint's config",
    cause: "the config states a variant of a setting (an activation, a norm, a router score, a rotation, a layer kind) that Ster's decoder for its family does not implement, so the decoder would compute another model than the weights were trained as",
    observed: SETTINGS_OBSERVED,
    remedy: "load a checkpoint of a variant Ster implements for the family (README.md lists the families Ster runs); `ster support report --description \"…\"` asks Ster's maintainers for this one",
    test: "tests/models/loading/load.sh::a_setting_ster_does_not_implement_is_refused",
};

pub static SETTINGS_CONTRADICTORY: Cause = Cause {
    id: "settings-contradictory",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the decoder's settings from the checkpoint's config",
    cause: "two settings of the config disagree (a per-layer list of another length than the layers, heads a group count does not divide, widths that do not fit one another), so no decoder can be built that follows both",
    observed: SETTINGS_OBSERVED,
    remedy: "correct the key the observed sentence names to the value the checkpoint's publisher states: the weights' shapes show which of the two settings is right",
    test: "tests/models/loading/settings.sh::an_indexer_narrower_than_the_rotation_is_refused",
};

pub static SETTINGS_OUT_OF_RANGE: Cause = Cause {
    id: "settings-out-of-range",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the decoder's settings from the checkpoint's config",
    cause: "a setting of the config holds a value outside the range the decoder runs (zero where at least one is needed, a width past the head, a layer index past the layers, a scale that is not positive)",
    observed: SETTINGS_OBSERVED,
    remedy: "state the setting within the range the observed sentence names, as the checkpoint's publisher does",
    test: "tests/models/loading/settings.sh::a_rotation_wider_than_the_head_is_refused",
};

pub static SETTINGS_INCOMPLETE: Cause = Cause {
    id: "settings-incomplete",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the decoder's settings from the checkpoint's config",
    cause: "the config turns on a part of the decoder (latent attention, a state-space mixer, an expert router, a scaled rotation) and leaves out a setting that part is sized by, and the family gives that setting no default",
    observed: SETTINGS_OBSERVED,
    remedy: "state the settings the observed sentence names, as the checkpoint's publisher does; a config cut short or written by hand leaves them out",
    test: "tests/models/loading/settings.sh::an_indexer_without_a_latent_query_is_refused",
};

pub static SETTINGS_MALFORMED: Cause = Cause {
    id: "settings-malformed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the decoder's settings from the checkpoint's config",
    cause: "a setting of the config is not in the form its family spells it (a per-layer mark other than the ones the family names, a list holding something other than numbers, an entry that is not a token id or an object)",
    observed: SETTINGS_OBSERVED,
    remedy: "state the setting in the form the observed sentence names, as the checkpoint's publisher does",
    test: "tests/models/loading/settings.sh::a_layer_mark_out_of_form_is_refused",
};

/// Reading a checkpoint's config into the decoder Ster builds for it.
pub static CONFIG: Operation = Operation {
    id: "ster.model.config",
    entries: &[],
    uses: &[&super::super::file::FILE],
    summary: "read a checkpoint's config (config.json, or Mistral's params.json put in its shape), choose the decoder family its model_type names, and read every setting that decoder needs, refusing one it does not implement",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &NOT_OBJECT,
        &MODEL_TYPE_ABSENT,
        &MODEL_TYPE_NOT_TEXT,
        &FAMILY_UNKNOWN,
        &QUANTIZED,
        &KEY_ABSENT,
        &KEY_MISTYPED,
        &UNDECODABLE,
        &SETTINGS_UNIMPLEMENTED,
        &SETTINGS_CONTRADICTORY,
        &SETTINGS_OUT_OF_RANGE,
        &SETTINGS_INCOMPLETE,
        &SETTINGS_MALFORMED,
    ],
};
