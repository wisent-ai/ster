//! How reading and checking a fixture's declaration, Llama config and
//! tokenizer fails (src/fixture/spec.rs), before anything is written.

use wisent_errors::{Cause, Code};

const EXAMPLE: &str = "docs/examples/model/fixture.json, config.json and vocabulary.json show each field; run `ster model fixture create --from <declaration> --output <new directory>` again";

pub static INPUT_MISSING: Cause = Cause {
    id: "input-missing",
    code: Code::NotFound,
    usage: true,
    exit: None,
    failed: "reading fixture {role} {path}",
    cause: "no file is at {path}",
    observed: "path: {path} ({state})",
    remedy: "give the path of an existing fixture {role}; the config and tokenizer paths resolve relative to the declaration",
    test: "tests/models/fixture.sh::an_input_that_does_not_exist_is_refused",
};

pub static INPUT_UNREADABLE: Cause = Cause {
    id: "input-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading fixture {role} {path}",
    cause: "the file cannot be read: this account may not read it, or it is no regular file",
    observed: "path: {path} ({state}); error kind {kind}; the system answered: {error}",
    remedy: "give this account read access to {path} (chmod u+r), or name a regular file",
    test: "tests/models/fixture.sh::an_unreadable_input_is_refused",
};

pub static INPUT_NOT_JSON: Cause = Cause {
    id: "input-not-json",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading fixture {role} {path}",
    cause: "it is not JSON",
    observed: "the JSON reader stopped at line {line}, column {column}: {error}",
    remedy: EXAMPLE,
    test: "tests/models/fixture.sh::an_input_that_is_not_json_is_refused",
};

pub static INPUT_SHAPE: Cause = Cause {
    id: "input-shape",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading fixture {role} {path}",
    cause: "it is JSON, and a field the fixture {role} requires is missing, one it does not take is present, or one holds another type",
    observed: "at line {line}, column {column}: {error}",
    remedy: EXAMPLE,
    test: "tests/models/fixture.sh::an_input_of_another_shape_is_refused",
};

pub static MODEL_TYPE: Cause = Cause {
    id: "model-type-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture config {path}",
    cause: "the fixture tensor layout requires model_type=llama, and the config declares {model_type}",
    observed: "model_type: {model_type}",
    remedy: "set model_type to \"llama\" in {path}; the fixture writes Llama checkpoints only",
    test: "tests/models/fixture.sh::a_config_of_another_family_is_refused",
};

pub static MLP_BIAS: Cause = Cause {
    id: "mlp-bias-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture config {path}",
    cause: "the config declares mlp_bias=true, but Ster's Llama consumer does not apply feed-forward biases; no checkpoint was written",
    observed: "{path}: \"mlp_bias\": true",
    remedy: "set mlp_bias to false in {path}",
    test: "tests/models/fixture.sh::a_feed_forward_bias_is_refused",
};

pub static HEADS_UNGROUPED: Cause = Cause {
    id: "heads-ungrouped",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture config {path}",
    cause: "the {heads} attention heads do not split into whole groups over the {kv} key/value heads, which grouped-query attention requires",
    observed: "num_attention_heads: {heads}; num_key_value_heads: {kv}; remainder {remainder}",
    remedy: "set num_key_value_heads in {path} to a divisor of {heads}",
    test: "tests/models/fixture.sh::heads_that_do_not_group_are_refused",
};

pub static NOT_FINITE: Cause = Cause {
    id: "setting-not-finite",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture {role} {path}",
    cause: "{key} must be a finite number, and it is {value}",
    observed: "{key}: {value}",
    remedy: "give {key} in {path} a finite number",
    test: "tests/models/fixture.sh::a_setting_that_is_not_finite_is_refused",
};

pub static SCALE_NEGATIVE: Cause = Cause {
    id: "scale-negative",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "drawing fixture {field} from its normal distribution",
    cause: "its scale, a standard deviation, is negative",
    observed: "{field}: mean {mean}, scale {scale}; the sampler answered: {error}",
    remedy: "give {field}.scale in {path} zero or a positive number",
    test: "tests/models/fixture.sh::a_negative_scale_is_refused",
};

pub static BIAS_MISSING: Cause = Cause {
    id: "bias-missing",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture declaration {path}",
    cause: "initialization.bias is required because the config declares attention_bias=true, and the declaration gives none",
    observed: "{config}: \"attention_bias\": true; {path}: initialization.bias absent",
    remedy: "give initialization.bias in {path} a mean and scale, or set attention_bias to false in the config",
    test: "tests/models/fixture.sh::an_attention_bias_without_its_initialization_is_refused",
};

pub static PROMPT_BLANK: Cause = Cause {
    id: "prompt-blank",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture declaration {path}",
    cause: "verification.prompt is required: the checkpoint is verified by forwarding it, and it is blank",
    observed: "verification.prompt: {prompt}",
    remedy: "give verification.prompt in {path} a text the tokenizer spells",
    test: "tests/models/fixture.sh::a_blank_verification_prompt_is_refused",
};

pub static LAYERS_NONE: Cause = Cause {
    id: "layers-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture declaration {path}",
    cause: "verification.layers is required: the checkpoint is verified at those layers, and it lists none",
    observed: "verification.layers: []; the config declares {count} layers",
    remedy: "list in verification.layers of {path} at least one layer from 0 to {last}",
    test: "tests/models/fixture.sh::no_verification_layer_is_refused",
};

pub static LAYER_OUTSIDE: Cause = Cause {
    id: "layer-outside",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture declaration {path}",
    cause: "verification layer {layer} lies outside {count} declared layers",
    observed: "verification.layers: {layers}; num_hidden_layers: {count}",
    remedy: "list in verification.layers of {path} only layers from 0 to {last}",
    test: "tests/models/fixture.sh::a_verification_layer_past_the_model_is_refused",
};

pub static TOKENIZER_UNPARSED: Cause = Cause {
    id: "tokenizer-unparsed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading fixture tokenizer {path}",
    cause: "the tokenizers library does not read it as a tokenizer.json: it is not JSON, or its model, normalizer or pre-tokenizer is missing or unknown to the library",
    observed: "path: {path}; the library answered: {error}",
    remedy: "name a tokenizer.json a tokenizer saved, or declare one with a tokenizer declaration (docs/guide/models/tokenizers.md)",
    test: "tests/models/fixture.sh::a_tokenizer_the_library_does_not_read_is_refused",
};

pub static TOKENIZER_EMPTY: Cause = Cause {
    id: "tokenizer-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading fixture tokenizer {path}",
    cause: "it holds no token, so no text could be spelled for the checkpoint",
    observed: "path: {path}; tokens: 0",
    remedy: "name a tokenizer that holds the tokens of the verification prompt",
    test: "tests/models/fixture.sh::an_empty_tokenizer_is_refused",
};

pub static TOKEN_OUTSIDE: Cause = Cause {
    id: "token-outside",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking fixture tokenizer {path} against the config",
    cause: "token {token} has id {id}, outside config vocab_size {size}, so the embedding would hold no row for it",
    observed: "token {token}: id {id}; vocab_size: {size}; highest id: {highest}",
    remedy: "set vocab_size in the config to at least {needed}, or give a tokenizer whose ids stay below {size}",
    test: "tests/models/fixture.sh::a_token_past_the_vocabulary_is_refused",
};

pub static TOKENIZER_TOO_WIDE: Cause = Cause {
    id: "tokenizer-too-wide",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking compiled fixture tokenizer {path} against the config",
    cause: "the compiled tokenizer has {count} token identities, exceeding config vocab_size {size}",
    observed: "token identities: {count}; vocab_size: {size}",
    remedy: "set vocab_size in the config to at least {count}, or declare fewer tokens",
    test: "tests/models/fixture.sh::a_declared_tokenizer_past_the_vocabulary_is_refused",
};

pub static WIDTH_OVERFLOW: Cause = Cause {
    id: "width-overflow",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "sizing the {what} projections of fixture config {path}",
    cause: "head_dim {head_dim} times {heads} {what} heads overflows this host's {bits}-bit sizes",
    observed: "head_dim: {head_dim}; {what} heads: {heads}",
    remedy: "lower head_dim or the head count in {path}; a fixture is meant to be small",
    test: "tests/models/fixture.sh::a_width_past_the_host_is_refused",
};
