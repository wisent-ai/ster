//! How comparing steering artifacts fitted for different traits fails
//! (src/workflow/compare): how many artifacts and groups, which layers,
//! whether they come from one model, and whether each direction has a
//! length to measure an angle by.

use wisent_errors::{Cause, Code, Operation};

use super::super::steering::ARTIFACT;
use super::DOCS;

const FAILED_STEP: &str = "comparing the steering artifacts {labels}";

pub static ARTIFACTS_FEW: Cause = Cause {
    id: "artifacts-few",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "vector compare needs at least two artifacts and got {count}: a comparison is between directions",
    observed: "artifacts named: {count} ({labels})",
    remedy: "name two or more steering artifacts fitted on one model, such as ster vector compare calm.json formal.json --clusters 1",
    test: "tests/steering/workflow.sh::one_artifact_to_compare_is_refused",
};

pub static MODELS_DIFFER: Cause = Cause {
    id: "models-differ",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "vector compare compares directions of one model: {first} is {first_model} at width {first_width}, {other} is {other_model} at width {other_width}, and two models' residual streams share no basis",
    observed: "{first}: model {first_model}, hidden_size {first_width}; {other}: model {other_model}, hidden_size {other_width}",
    remedy: "compare artifacts fitted on one model; `ster vector inspect` shows each artifact's model and hidden_size",
    test: "tests/steering/workflow.sh::artifacts_of_two_models_are_refused",
};

pub static CLUSTERS_PAST: Cause = Cause {
    id: "clusters-past",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "--clusters {clusters} cannot group {count} artifacts: at most one group per artifact",
    observed: "--clusters (a request's clusters) {clusters}; artifacts: {count}",
    remedy: "give --clusters from 1 to {count}",
    test: "tests/steering/workflow.sh::more_groups_than_artifacts_are_refused",
};

pub static LAYER_UNCARRIED: Cause = Cause {
    id: "layer-uncarried",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "layer {layer} is not in {label}'s artifact, which carries layers {carried}, and every compared layer must be carried by every artifact",
    observed: "--layers {named}; layers every artifact carries: {common}",
    remedy: "name only layers every artifact carries ({common}), or leave --layers out to compare all of them",
    test: "tests/steering/workflow.sh::a_layer_an_artifact_lacks_is_refused",
};

pub static LAYERS_UNSHARED: Cause = Cause {
    id: "layers-unshared",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the artifacts share no layer to compare at: no layer carries a direction in every one of them",
    observed: "layers each artifact carries: {listed}",
    remedy: "train the artifacts at a common layer (ster vector train --layers), or compare only the ones that share one",
    test: "tests/steering/workflow.sh::artifacts_sharing_no_layer_are_refused",
};

pub static DIRECTION_LENGTHLESS: Cause = Cause {
    id: "direction-lengthless",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "{label}'s direction at layer {layer} has length {length}, so it has no angle to compare: every value is zero, or too small to measure",
    observed: "{label}: layer {layer}, {width} values, length {length}",
    remedy: "train the {label} vector again with ster vector train, or leave layer {layer} out with --layers",
    test: "tests/steering/workflow.sh::a_direction_without_length_is_refused",
};

/// `ster vector compare`: how alike several traits' directions are.
pub static COMPARE: Operation = Operation {
    id: "ster.vector.compare",
    entries: &["ster vector compare", "ster request vector/compare"],
    uses: &[&ARTIFACT],
    summary: "read two or more steering artifacts of one model, compare their unit directions at every shared layer, and cut them into the groups named",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &ARTIFACTS_FEW,
        &MODELS_DIFFER,
        &CLUSTERS_PAST,
        &LAYER_UNCARRIED,
        &LAYERS_UNSHARED,
        &DIRECTION_LENGTHLESS,
    ],
};
