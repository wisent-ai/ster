//! How fitting a direction on the two sides of the pairs, drawing a random
//! one, and scoring one fails (src/representation.rs): the estimator
//! `--method` names and the seed a random one is drawn with, the reads it is
//! fitted on, and the direction it produces. `{step}` says which fit it was
//! (`fitting caa at layer 1 on 4 pairs`).

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

pub static METHOD_UNKNOWN: Cause = Cause {
    id: "method-unknown",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the direction estimator --method {value}",
    cause: "unknown training method \"{value}\": Ster fits a direction by contrastive activation addition (caa), the leading principal direction (pca) or a logistic probe (logistic), or draws a seeded control direction (random), and by nothing else",
    observed: "--method (a request's method) \"{value}\"; expected caa, pca, logistic, or random",
    remedy: "give --method caa, pca, logistic or random",
    test: "tests/steering/vectors/layers.sh::an_unknown_method_is_refused",
};

pub static SEED_MISSING: Cause = Cause {
    id: "seed-missing",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "drawing a random direction",
    cause: "--method random draws its direction from a seed, and none was given; Ster assumes no seed",
    observed: "--method random without --seed (a request and ster vector curve take no seed)",
    remedy: "give ster vector train --method random a --seed, such as --seed 1, or fit the direction with caa, pca or logistic",
    test: "tests/steering/vectors/layers.sh::a_random_method_without_a_seed_is_refused",
};

pub static SEED_UNUSED: Cause = Cause {
    id: "seed-unused",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "choosing the direction estimator --method {method}",
    cause: "--seed {seed} is read only by --method random, and {method} fits its direction from the pairs with no seed",
    observed: "--method {method} --seed {seed}",
    remedy: "leave --seed out, or draw a control direction with --method random --seed {seed}",
    test: "tests/steering/vectors/layers.sh::a_seed_for_a_fitted_method_is_refused",
};

pub static EXAMPLES_UNEQUAL: Cause = Cause {
    id: "examples-unequal",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "training requires the same non-zero number of positive and negative examples; got {positive} positive and {negative} negative",
    observed: "positive reads: {positive}; negative reads: {negative}",
    remedy: "give a pair set where every pair has both sides; `ster pairs inspect` counts its pairs",
    test: "tests/steering/vectors/direction.sh::sides_of_unequal_counts_are_refused",
};

pub static READ_EMPTY: Cause = Cause {
    id: "read-empty",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the first {side} read holds no value, so there is no space to fit a direction in",
    observed: "{side} read 0: 0 values; the reads number {positive} positive and {negative} negative",
    remedy: "read the pairs with a model whose hidden_size is above 0",
    test: "tests/steering/vectors/direction.sh::reads_without_values_are_refused",
};

pub static READ_WIDTH: Cause = Cause {
    id: "read-width",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the {side} read {index} holds {found} values where the first read holds {width}, and every read of one layer has one width",
    observed: "{side} read {index}: {found} values; the first positive read: {width} values",
    remedy: "read every pair through one model at one layer",
    test: "tests/steering/vectors/direction.sh::reads_of_two_widths_are_refused",
};

pub static READ_NOT_FINITE: Cause = Cause {
    id: "read-not-finite",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the {side} read {index} holds a value that is not finite: the forward pass overflowed at this precision, or the weights hold one",
    observed: "{side} read {index}: value {position} of {width} is {value}",
    remedy: "read the pairs at a wider precision (--precision f32), or check the checkpoint's weights are finite",
    test: "tests/steering/vectors/direction.sh::reads_that_are_not_finite_are_refused",
};

pub static DIRECTION_WIDTH: Cause = Cause {
    id: "direction-width",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "direction width {direction} does not match activation width {width}: the direction was fitted on a model of another width",
    observed: "direction: {direction} values; each read: {width} values",
    remedy: "score the artifact with the model it was fitted on; `ster vector inspect` shows its model and width",
    test: "tests/steering/vectors/direction.sh::a_direction_of_another_width_is_refused",
};

pub static DIRECTION_DEGENERATE: Cause = Cause {
    id: "direction-degenerate",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "training produced a zero or non-finite direction: its length is {length}, so the two sides read alike at this layer, or the fit overflowed",
    observed: "direction length {length} over {width} values; {positive} pairs",
    remedy: "give pairs whose two sides differ in the trait, read another layer, or fit with another --method",
    test: "tests/steering/vectors/direction.sh::sides_that_read_alike_are_refused",
};

/// Fitting a direction on the reads of the pairs' two sides, or drawing a
/// random one, and scoring one on them.
pub static DIRECTION: Operation = Operation {
    id: "ster.vector.direction",
    entries: &[],
    uses: &[],
    summary: "fit a unit direction on one layer's reads of the pairs' positive and negative sides by caa, pca or logistic, or draw a seeded random one at their width, and score a direction by how it orders them",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &METHOD_UNKNOWN,
        &SEED_MISSING,
        &SEED_UNUSED,
        &EXAMPLES_UNEQUAL,
        &READ_EMPTY,
        &READ_WIDTH,
        &READ_NOT_FINITE,
        &DIRECTION_WIDTH,
        &DIRECTION_DEGENERATE,
    ],
};
