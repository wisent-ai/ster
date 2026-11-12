//! How LoRA adapters fail (src/lora): the spec a trainer is given, and the
//! artifact a trained adapter travels in — its safetensors file, the JSON
//! sidecar beside it, and the tensors it must hold.

use wisent_errors::{Cause, Code, Operation};

/// docs/guide/errors/lora.md, where these functionalities are listed.
const DOCS: &str = page!("lora");

pub mod artifact;

pub static TARGET_UNKNOWN: Cause = Cause {
    id: "target-unknown",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the adapter targets",
    cause: "unknown adapter target {target}; expected one of query, key, value, output, gate, up, down",
    observed: "--targets {targets}",
    remedy: "name targets among query, key, value, output, gate, up and down (or q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), comma-separated",
    test: "tests/tuning/runs/adapters.sh::an_unknown_target_is_refused_before_the_model_loads",
};

pub static TARGETS_NONE: Cause = Cause {
    id: "targets-none",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the adapter targets",
    cause: "adapter spec names no targets; choose at least one of query, key, value, output, gate, up, down",
    observed: "--targets {targets}",
    remedy: "give --targets at least one projection, such as query,value",
    test: "tests/tuning/runs/adapters.sh::no_target_is_refused_before_the_model_loads",
};

pub static TARGET_TWICE: Cause = Cause {
    id: "target-twice",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the adapter spec",
    cause: "adapter spec names target {target} twice",
    observed: "targets: {targets}",
    remedy: "name each target once",
    test: "tests/tuning/runs/adapters.sh::a_target_named_twice_by_a_request_is_refused",
};

pub static RANK_ZERO: Cause = Cause {
    id: "rank-zero",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the adapter spec",
    cause: "adapter rank 0 trains nothing; choose a rank of at least 1",
    observed: "--rank {rank}",
    remedy: "give --rank at least 1, such as 8",
    test: "tests/tuning/runs/adapters.sh::a_rank_of_zero_is_refused_before_the_model_loads",
};

pub static ALPHA_UNUSABLE: Cause = Cause {
    id: "alpha-unusable",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "checking the adapter spec",
    cause: "adapter alpha {alpha} is not a positive finite number",
    observed: "--alpha {alpha}",
    remedy: "give --alpha a finite number above zero; twice the rank is a common choice",
    test: "tests/tuning/runs/adapters.sh::an_alpha_of_zero_is_refused_before_the_model_loads",
};

pub static LAYER_PAST: Cause = Cause {
    id: "layer-past",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "placing the adapters on the model's layers",
    cause: "adapter spec names layer {layer}, but the model has {count} layers",
    observed: "--layers {layers}",
    remedy: "give --layers indexes from 0 to {last}, or all",
    test: "tests/tuning/runs/adapters.sh::a_layer_past_the_model_is_refused",
};

pub static LAYERS_NONE: Cause = Cause {
    id: "layers-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "placing the adapters on the model's layers",
    cause: "the model's config states {count} layers, so it holds no projection an adapter can be placed on",
    observed: "num_hidden_layers: {count}",
    remedy: "train on a checkpoint whose config.json states num_hidden_layers of at least 1",
    test: "tests/tuning/runs/adapters.sh::a_model_without_layers_is_refused",
};

/// The rank, alpha, targets and layers a trainer is given.
pub static SPEC: Operation = Operation {
    id: "ster.lora.spec",
    entries: &[],
    uses: &[],
    summary: "read the rank, alpha, projections and layers of the LoRA adapters a run creates; every check but the layer count is made before a model loads",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &TARGET_UNKNOWN,
        &TARGETS_NONE,
        &TARGET_TWICE,
        &RANK_ZERO,
        &ALPHA_UNUSABLE,
        &LAYER_PAST,
        &LAYERS_NONE,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&SPEC, &artifact::ARTIFACT];
