//! How choosing the layers a run reads fails (`workflow::parse_layers`):
//! `all`, a comma list or a half-open range such as `8..16`, as `--layers`
//! (a request's `layers`) names them, checked against the model's layers
//! once its config is read. `{model_layers}` says how many layers the model
//! has, or that they are counted only once the weights are mapped (`ster
//! tune`), and `{example}` is a range the model has.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

const FAILED_STEP: &str = "choosing the layers --layers {selection} names";

pub static MODEL_LAYERLESS: Cause = Cause {
    id: "model-layerless",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: FAILED_STEP,
    cause: "the model's config names no decoder layer (num_hidden_layers is 0), so there is no layer to read or steer",
    observed: "--layers {selection}; the model has 0 layers",
    remedy: "name a checkpoint whose config.json gives num_hidden_layers above 0",
    test: "tests/steering/vectors/layers.sh::a_model_without_layers_is_refused",
};

pub static RANGE_BOUND_UNPARSED: Cause = Cause {
    id: "range-bound-unparsed",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "the range {segment} has a bound, \"{bound}\", that is not a whole number of zero or more, and a range is START..END of two layer numbers",
    observed: "--layers {selection}; segment {segment}; bound \"{bound}\": {error}; {model_layers}",
    remedy: "write the range as two layer numbers, such as {example}, or name the layers as a comma list",
    test: "tests/steering/vectors/layers.sh::a_range_bound_that_is_not_a_number_is_refused",
};

pub static RANGE_EMPTY: Cause = Cause {
    id: "range-empty",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "layer range {segment} must have start < end: a range is half-open, so START..END reads START up to END-1 and {start}..{end} reads no layer",
    observed: "--layers {selection}; segment {segment}: start {start}, end {end}; {model_layers}",
    remedy: "give the smaller layer first, such as {end}..{after}, or name one layer as {start}",
    test: "tests/steering/vectors/layers.sh::a_range_that_holds_no_layer_is_refused",
};

pub static LAYER_UNPARSED: Cause = Cause {
    id: "layer-unparsed",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "\"{segment}\" is not a layer number: --layers takes `all`, whole numbers of zero or more separated by commas, or a range START..END",
    observed: "--layers {selection}; segment \"{segment}\": {error}; {model_layers}",
    remedy: "name the layers as all, a comma list such as 0,1 or a range such as {example}",
    test: "tests/steering/vectors/layers.sh::a_layer_that_is_not_a_number_is_refused",
};

pub static SELECTION_EMPTY: Cause = Cause {
    id: "selection-empty",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "no layers selected: the selection holds only commas and spaces, so it names no layer",
    observed: "--layers \"{selection}\"; {model_layers}",
    remedy: "name at least one layer, such as {example}, or every layer with all",
    test: "tests/steering/vectors/layers.sh::a_selection_naming_no_layer_is_refused",
};

pub static LAYER_PAST: Cause = Cause {
    id: "layer-past",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: FAILED_STEP,
    cause: "layer {layer} is outside the model's {count} layers, counted from the first, so the model has layers 0 to {last}",
    observed: "--layers {selection}; every layer named past the model: {past}",
    remedy: "name layers from 0 to {last}, such as {example}, or all",
    test: "tests/steering/vectors/layers.sh::a_layer_past_the_model_is_refused",
};

/// Turning a `--layers` selection into the model's layer numbers.
pub static LAYERS: Operation = Operation {
    id: "ster.vector.layers",
    entries: &[],
    uses: &[],
    summary: "read a --layers selection (all, a comma list or a half-open range) as layer numbers the model has",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &MODEL_LAYERLESS,
        &RANGE_BOUND_UNPARSED,
        &RANGE_EMPTY,
        &LAYER_UNPARSED,
        &SELECTION_EMPTY,
        &LAYER_PAST,
    ],
};
