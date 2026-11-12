//! How a trained adapter's artifact fails (src/lora/artifact): its
//! safetensors file, the JSON sidecar beside it, and the tensors it holds.
//! Either file failing to be read or written, or not being JSON or
//! safetensors of its form, is the `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;
use crate::failures::file::FILE;

/// What an adapter's sidecar is, for the `ster.file` refusals that name it.
pub const SIDECAR_FORM: &str = "the JSON sidecar a `ster tune` command writes beside the adapter's .safetensors file";

pub static ADAPTER_MISSING: Cause = Cause {
    id: "adapter-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading adapter {path}",
    cause: "no file is there: nothing at {path} is the adapter named",
    observed: "path: {path} ({state}); its directory {parent}",
    remedy: "name the .safetensors file a ster tune command wrote, with its .json sidecar beside it; `ls -ln {parent}` shows what that directory holds",
    test: "tests/tuning/adapters.sh::a_missing_adapter_is_refused",
};

pub static SIDECAR_MISSING: Cause = Cause {
    id: "sidecar-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading adapter {path}",
    cause: "adapter {path} has no sidecar at {sidecar}; the pair is written together and must travel together",
    observed: "missing: {sidecar}",
    remedy: "copy {sidecar} beside the weights from where the adapter was trained",
    test: "tests/tuning/adapters.sh::an_adapter_without_its_sidecar_is_refused",
};

pub static STEERING_GIVEN: Cause = Cause {
    id: "steering-given",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading adapter {path}",
    cause: "{sidecar} is a steering artifact, not a LoRA adapter sidecar: it carries trait_name and vectors where an adapter sidecar carries rank and targets",
    observed: "{sidecar}: a steering artifact",
    remedy: "pass a steering artifact with --vector and a LoRA adapter with --adapter",
    test: "tests/tuning/adapters.sh::a_steering_artifact_given_as_an_adapter_is_refused",
};

pub static SCHEMA_UNSUPPORTED: Cause = Cause {
    id: "schema-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "adapter artifact schema {schema} is unsupported; this Ster build reads schema {supported}",
    observed: "schema_version: {schema}",
    remedy: "read it with the Ster release that wrote it, or train the adapter again with this one",
    test: "tests/tuning/adapters.sh::a_sidecar_of_another_schema_is_refused",
};

pub static PRODUCT_FOREIGN: Cause = Cause {
    id: "product-foreign",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "adapter artifact belongs to product {product}, not Ster",
    observed: "product: {product}",
    remedy: "read it with the product that wrote it; Ster reads the adapters ster tune writes",
    test: "tests/tuning/adapters.sh::a_sidecar_of_another_product_is_refused",
};

/// What every sidecar declaration refusal tells its reader to do.
const DECLARED_REMEDY: &str = "use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again";

pub static RANK_ZERO: Cause = Cause {
    id: "rank-zero",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "the sidecar declares rank 0, and an adapter of rank 0 stores no update: every factor pair is zero columns wide, which no Ster trainer writes",
    observed: "rank: 0",
    remedy: DECLARED_REMEDY,
    test: "tests/tuning/sidecar.sh::a_sidecar_declaring_rank_zero_is_refused",
};

pub static ALPHA_INVALID: Cause = Cause {
    id: "alpha-invalid",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "the sidecar declares an alpha that is not a positive finite number, so the update's scale, alpha over rank, is zero, negative or not a number",
    observed: "alpha: {alpha}; rank: {rank}",
    remedy: DECLARED_REMEDY,
    test: "tests/tuning/sidecar.sh::a_sidecar_declaring_alpha_zero_is_refused",
};

pub static HIDDEN_SIZE_ZERO: Cause = Cause {
    id: "hidden-size-zero",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "the sidecar declares hidden size 0, and an adapter is trained on a model whose residual stream is at least one wide",
    observed: "hidden_size: 0; model: {model}",
    remedy: DECLARED_REMEDY,
    test: "tests/tuning/sidecar.sh::a_sidecar_declaring_hidden_size_zero_is_refused",
};

pub static TARGETS_NONE: Cause = Cause {
    id: "targets-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "the sidecar names no target projection, so the adapter adapts nothing",
    observed: "targets: []; layers: {layers}",
    remedy: DECLARED_REMEDY,
    test: "tests/tuning/sidecar.sh::a_sidecar_naming_no_target_is_refused",
};

pub static LAYERS_NONE: Cause = Cause {
    id: "layers-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "the sidecar names no layer, so the adapter adapts nothing",
    observed: "layers: []; targets: {targets}",
    remedy: DECLARED_REMEDY,
    test: "tests/tuning/sidecar.sh::a_sidecar_naming_no_layer_is_refused",
};

pub static TARGET_REPEATED: Cause = Cause {
    id: "target-repeated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "the sidecar names target {target} twice, and a projection holds one factor pair per layer, so the two entries would name the same tensors",
    observed: "targets: {targets}",
    remedy: DECLARED_REMEDY,
    test: "tests/tuning/sidecar.sh::a_sidecar_naming_a_target_twice_is_refused",
};

pub static LAYER_REPEATED: Cause = Cause {
    id: "layer-repeated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "the sidecar names layer {layer} twice, and a layer holds one factor pair per target, so the two entries would name the same tensors",
    observed: "layers: {layers}",
    remedy: DECLARED_REMEDY,
    test: "tests/tuning/sidecar.sh::a_sidecar_naming_a_layer_twice_is_refused",
};

pub static TENSOR_ABSENT: Cause = Cause {
    id: "tensor-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "adapter artifact is missing tensor {name}, which its sidecar's targets and layers need",
    observed: "missing: {name}",
    remedy: "use the weights and the sidecar a ster tune command wrote together",
    test: "tests/tuning/adapters.sh::a_sidecar_naming_a_layer_the_weights_lack_is_refused",
};

pub static TENSOR_SHAPE: Cause = Cause {
    id: "tensor-shape",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "adapter tensor {name} has {found}, expected {expected}",
    observed: "{name}: {shape}",
    remedy: "use the weights and the sidecar a ster tune command wrote together, for the model it was trained on",
    test: "tests/tuning/adapters.sh::a_sidecar_of_another_rank_is_refused",
};

pub static TENSOR_NOT_FINITE: Cause = Cause {
    id: "tensor-not-finite",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "adapter tensor {name} contains a non-finite value, which would carry NaN through every forward pass",
    observed: "{name}: {count} non-finite values",
    remedy: "train the adapter again with a lower --learning-rate; a run that diverged writes infinite factors",
    test: "tests/tuning/adapters.sh::a_non_finite_factor_is_refused",
};

pub static HEAD_MISSING: Cause = Cause {
    id: "head-missing",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "reward artifact is missing tensor reward.head, which is the head it exists to carry",
    observed: "kind: reward, tensors: {count}",
    remedy: "use the weights ster tune reward wrote beside this sidecar",
    test: "tests/tuning/adapters.sh::a_reward_sidecar_without_its_head_is_refused",
};

pub static HEAD_UNEXPECTED: Cause = Cause {
    id: "head-unexpected",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the adapter artifact",
    cause: "adapter artifact carries a reward.head tensor but declares kind adapter",
    observed: "kind: adapter, with reward.head of shape {shape} among {count} tensors; trained for {model}",
    remedy: "use the sidecar ster tune reward wrote beside these weights",
    test: "tests/tuning/adapters.sh::an_adapter_sidecar_over_reward_weights_is_refused",
};

pub static WIDTH_UNFIT: Cause = Cause {
    id: "width-unfit",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "fitting the adapter to the model's projections",
    cause: "adapter tensor {name} has shape {found}, expected {expected} for the model projection",
    observed: "{name}: {found}",
    remedy: "use the adapter with the model it was trained on",
    test: "tests/tuning/adapters.sh::an_adapter_of_another_projection_width_is_refused",
};

/// Reading, checking and writing one adapter artifact.
pub static ARTIFACT: Operation = Operation {
    id: "ster.lora.artifact",
    entries: &["ster tune inspect", "ster request tune/inspect"],
    uses: &[&FILE],
    summary: "read a LoRA adapter's safetensors weights and JSON sidecar, check every tensor the sidecar declares is there, shaped and finite, and write one back",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &ADAPTER_MISSING,
        &SIDECAR_MISSING,
        &STEERING_GIVEN,
        &SCHEMA_UNSUPPORTED,
        &PRODUCT_FOREIGN,
        &RANK_ZERO,
        &ALPHA_INVALID,
        &HIDDEN_SIZE_ZERO,
        &TARGETS_NONE,
        &LAYERS_NONE,
        &TARGET_REPEATED,
        &LAYER_REPEATED,
        &TENSOR_ABSENT,
        &TENSOR_SHAPE,
        &TENSOR_NOT_FINITE,
        &HEAD_MISSING,
        &HEAD_UNEXPECTED,
        &WIDTH_UNFIT,
    ],
};
