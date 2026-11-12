//! How steering a loaded model with an artifact fails
//! (src/runtime/infer/steer.rs), and the layer every artifact and every
//! single-layer read names (src/runtime/mod.rs).

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

pub static LAYER_OUTSIDE: Cause = Cause {
    id: "layer-outside",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading layer {layer} of the model",
    cause: "layer {layer} is outside the model's {count} layers",
    observed: "layers named: {layers}; the model has layers 0 to {last}",
    remedy: "name a layer from 0 to {last}, or use the artifact with the model it was trained on",
    test: "tests/runtime/steer.sh::a_vector_past_the_model_layers_is_refused",
};

/// A layer every artifact and every single-layer read is checked against.
pub static LAYERS: Operation = Operation {
    id: "ster.runtime.layers",
    entries: &[],
    uses: &[],
    summary: "check that every layer an artifact or a read names is one of the model's",
    docs: DOCS,
    warnings: &[],
    causes: &[&LAYER_OUTSIDE],
};

pub static MODEL_OTHER: Cause = Cause {
    id: "model-other",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "steering {model} with the {trait_name} artifact",
    cause: "the artifact was trained for model {trained}, current model is {model}",
    observed: "trained for: {trained}",
    remedy: "steer the model the artifact was trained for, --model {trained}, or train a vector for this one with ster vector train",
    test: "tests/runtime/steer.sh::a_vector_for_another_model_is_refused",
};

pub static WIDTH_OTHER: Cause = Cause {
    id: "width-other",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "steering {model} with the {trait_name} artifact",
    cause: "artifact width {trained} does not match model width {width}",
    observed: "artifact hidden_size: {trained}; model hidden_size: {width}",
    remedy: "steer the model the artifact was trained for; a vector fits only the width it was trained at",
    test: "tests/runtime/steer.sh::a_vector_of_another_width_is_refused",
};

pub static STRENGTH_NOT_FINITE: Cause = Cause {
    id: "strength-not-finite",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "scaling the steering vectors",
    cause: "steering strength {strength} is not finite",
    observed: "--strength {strengths}",
    remedy: "give every --strength a finite number, such as 4",
    test: "tests/runtime/steer.sh::a_strength_that_is_not_finite_is_refused",
};

pub static STRENGTHS_ZERO: Cause = Cause {
    id: "strengths-zero",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "scaling the steering vectors",
    cause: "every steering strength is zero, so the vectors would steer nothing",
    observed: "--strength {strengths}",
    remedy: "give at least one --strength other than zero, or leave --vector out to generate unsteered",
    test: "tests/runtime/steer.sh::strengths_of_zero_are_refused",
};

/// Adding steering artifacts to a loaded model's residual stream.
pub static STEER: Operation = Operation {
    id: "ster.runtime.steer",
    entries: &[],
    uses: &[&LAYERS],
    summary: "check that each steering artifact was trained for this model at its width and layers, and scale every vector by its strength",
    docs: DOCS,
    warnings: &[],
    causes: &[&MODEL_OTHER, &WIDTH_OTHER, &STRENGTH_NOT_FINITE, &STRENGTHS_ZERO],
};
