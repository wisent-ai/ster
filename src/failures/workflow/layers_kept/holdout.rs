//! How the commands that read every pair through the model and fit a
//! direction fail: `vector train`, `vector optimize`, `vector curve` and
//! `vector evaluate`, and the held-out fraction `optimize` and `curve` rank
//! on (src/workflow/{train,optimize,curve,evaluate}.rs).

use wisent_errors::{Cause, Code, Operation};

use super::super::chat::TEMPLATE;
use super::super::file::FILE;
use super::super::model::weights::LOAD;
use super::super::pairs::FILE as PAIRS;
use super::super::runtime::encode::ENCODE;
use super::super::runtime::steer::STEER;
use super::super::steering::ARTIFACT;
use super::super::tune::strength::STRENGTH;
use super::super::workspace::ACTIVE;
use super::DOCS;
use super::direction::DIRECTION;
use super::layers::LAYERS;

pub static FRACTION_OUTSIDE: Cause = Cause {
    id: "fraction-outside",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "holding out --holdout {holdout} of the {total} pairs of the \"{trait}\" pair set",
    cause: "the held-out fraction must be above zero and below one: it is the share of the pairs every candidate is ranked on, and the rest are fitted on",
    observed: "--holdout (a request's holdout) {holdout}; the pair set holds {total} pairs",
    remedy: "give --holdout a fraction above 0 and below 1 that leaves a pair on each side, such as {example}",
    test: "tests/steering/vectors/holdout.sh::a_fraction_outside_zero_and_one_is_refused",
};

pub static SIDE_EMPTY: Cause = Cause {
    id: "side-empty",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "holding out --holdout {holdout} of the {total} pairs of the \"{trait}\" pair set",
    cause: "a held-out fraction of {holdout} over {total} contrastive pairs leaves {fit} pairs to fit and {held} to rank on; both need at least one",
    observed: "--holdout {holdout} × {total} pairs rounds to {held} held out; {fit} left to fit on",
    remedy: "give --holdout a fraction that leaves a pair on each side, such as {example}, or add pairs with `ster pairs add`",
    test: "tests/steering/vectors/holdout.sh::a_fraction_that_empties_a_side_is_refused",
};

pub static SIZE_PAST: Cause = Cause {
    id: "size-past",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the sizes vector curve fits {method} on",
    cause: "a size of {size} pairs does not fit in the {fit} pairs left to fit on after holding out {held}",
    observed: "--sizes {sizes}; sizes past the fitting part: {past}; the \"{trait}\" pair set holds {total} pairs",
    remedy: "name sizes of at most {fit} pairs, give a smaller --holdout, or add pairs with `ster pairs add`",
    test: "tests/steering/vectors/holdout.sh::a_size_past_the_fitting_part_is_refused",
};

/// Holding out the last fraction of a pair set to rank candidates on.
pub static HOLDOUT: Operation = Operation {
    id: "ster.vector.holdout",
    entries: &[],
    uses: &[],
    summary: "cut a pair set into the pairs every candidate is fitted on and the held-out last fraction it is ranked on, each holding a pair",
    docs: DOCS,
    warnings: &[],
    causes: &[&FRACTION_OUTSIDE, &SIDE_EMPTY],
};

/// `ster vector train`: one direction per layer fitted on every pair.
pub static TRAIN: Operation = Operation {
    id: "ster.vector.train",
    entries: &["ster vector train", "ster request vector/train"],
    uses: &[&ACTIVE, &PAIRS, &LOAD, &TEMPLATE, &LAYERS, &ENCODE, &DIRECTION, &ARTIFACT, &FILE],
    summary: "read both sides of every pair through the model at the layers named, fit one direction per layer by the method named, and write the steering artifact",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// `ster vector optimize`: every layer and method ranked on a holdout.
pub static OPTIMIZE: Operation = Operation {
    id: "ster.vector.optimize",
    entries: &["ster vector optimize", "ster request vector/optimize"],
    uses: &[&ACTIVE, &PAIRS, &LOAD, &TEMPLATE, &LAYERS, &HOLDOUT, &ENCODE, &DIRECTION, &ARTIFACT, &FILE],
    summary: "fit every layer and method on part of the pairs, rank each on the held-out rest, refit the winner on every pair and write it with the ranking",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// `ster vector curve`: holdout accuracy against the number of pairs fitted.
pub static CURVE: Operation = Operation {
    id: "ster.vector.curve",
    entries: &["ster vector curve", "ster request vector/curve"],
    uses: &[&ACTIVE, &PAIRS, &LOAD, &TEMPLATE, &LAYERS, &HOLDOUT, &ENCODE, &DIRECTION],
    summary: "fit one method at every layer on each named number of pairs and score each fit on the same held-out pairs",
    docs: DOCS,
    warnings: &[],
    causes: &[&SIZE_PAST],
};

/// `ster vector evaluate`: how well each layer's direction orders the pairs.
pub static EVALUATE: Operation = Operation {
    id: "ster.vector.evaluate",
    entries: &["ster vector evaluate", "ster request vector/evaluate"],
    uses: &[&ACTIVE, &PAIRS, &ARTIFACT, &LOAD, &TEMPLATE, &STEER, &ENCODE, &DIRECTION, &STRENGTH],
    summary: "check the artifact fits the model, read both sides of every pair at the layers it carries, and score how each layer's direction orders them",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};
