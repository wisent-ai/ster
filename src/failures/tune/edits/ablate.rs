//! How writing a steering direction out of a checkpoint's weights fails
//! (src/tune/artifact/ablate.rs), beyond fitting the artifact to the
//! checkpoint and writing the result.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;
use super::rewrite::{FIT, WRITE};

pub static ABLATION_STRENGTH: Cause = Cause {
    id: "ablation-strength",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "removing the direction of {path} from {model}",
    cause: "ablation strength {strength} changes nothing or nothing sensible; it must be a finite, non-zero number",
    observed: "--strength {strength}",
    remedy: "give --strength a finite, non-zero number: 1 removes the direction's whole component",
    test: "tests/tuning/scales/rewrite.sh::an_ablation_strength_of_zero_is_refused",
};

pub static DIRECTION_EMPTY: Cause = Cause {
    id: "direction-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "removing the direction at layer {layer} of {path}",
    cause: "the direction at layer {layer} has length {length}, so there is nothing to remove",
    observed: "length of layer {layer}: {length}",
    remedy: "train the direction again with ster vector train; a direction of length zero says the pairs did not separate",
    test: "tests/tuning/scales/rewrite.sh::a_direction_of_length_zero_is_refused",
};

pub static WRITE_WIDTH: Cause = Cause {
    id: "write-width",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "removing the direction at layer {layer} of {path}",
    cause: "{name} writes {outputs} outputs where the residual stream is {width} wide",
    observed: "{name}: {outputs} outputs",
    remedy: "name the checkpoint the direction was trained on; its residual-stream writes are as wide as its hidden_size",
    test: "tests/tuning/scales/rewrite.sh::a_write_of_another_width_is_refused",
};

/// `ster vector ablate`: a direction removed from the weights.
pub static ABLATE: Operation = Operation {
    id: "ster.tune.ablate",
    entries: &["ster vector ablate", "ster request vector/ablate"],
    uses: &[&FIT, &WRITE],
    summary: "rewrite a checkpoint's attention-output and MLP-down projections so they no longer write a steering direction, and save it as a standalone checkpoint",
    docs: DOCS,
    warnings: &[],
    causes: &[&ABLATION_STRENGTH, &DIRECTION_EMPTY, &WRITE_WIDTH],
};
