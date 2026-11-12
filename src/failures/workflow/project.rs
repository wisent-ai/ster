//! How projecting a pair set's states along an artifact and scoring a text
//! token by token fail (src/workflow/project): the reads to project onto,
//! the text to score, and the artifact's direction at the layer named.

use wisent_errors::{Cause, Code, Operation};

use super::super::chat::TEMPLATE;
use super::super::file::FILE;
use super::super::model::weights::LOAD;
use super::super::pairs::FILE as PAIRS;
use super::super::runtime::encode::ENCODE;
use super::super::runtime::steer::{LAYERS as MODEL_LAYERS, STEER};
use super::super::steering::ARTIFACT;
use super::super::workspace::ACTIVE;
use super::DOCS;

pub static READS_INVARIANT: Cause = Cause {
    id: "reads-invariant",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "projecting the {pairs} pairs of the \"{trait}\" pair set at layer {layer}",
    cause: "the unsteered reads at layer {layer} do not vary, so they have no component to project onto: every side reads the same state there",
    observed: "{reads} unsteered reads at layer {layer}, {components} component(s) asked for, none with variance",
    remedy: "give pairs whose sides differ, or read another --layer; layer 0 reads little more than the token embeddings",
    test: "tests/steering/workflow.sh::reads_that_do_not_vary_are_refused",
};

/// `ster vector project`: where an artifact moves a pair set's states.
pub static PROJECT: Operation = Operation {
    id: "ster.vector.project",
    entries: &["ster vector project", "ster request vector/project"],
    uses: &[&ACTIVE, &PAIRS, &ARTIFACT, &LOAD, &TEMPLATE, &STEER, &MODEL_LAYERS, &ENCODE],
    summary: "read every side of every pair at one layer unsteered and steered, and project both onto the principal components of the unsteered reads",
    docs: DOCS,
    warnings: &[],
    causes: &[&READS_INVARIANT],
};

const SCORE_STEP: &str = "scoring a text along the {trait} direction at layer {layer}";

pub static INPUT_ABSENT: Cause = Cause {
    id: "input-absent",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the text to score",
    cause: "give the text to score with --input or --input-file: the command scores one text and none was named",
    observed: "neither --input nor --input-file given; --vector {vector} --layer {layer}",
    remedy: "give --input TEXT, or --input-file FILE holding the text",
    test: "tests/steering/score.sh::no_text_is_refused",
};

pub static INPUT_TWICE: Cause = Cause {
    id: "input-twice",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the text to score",
    cause: "--input cannot be used with --input-file: give the text once, so it is clear which one is scored",
    observed: "--input ({characters} characters) and --input-file {path} both given",
    remedy: "keep --input or --input-file, not both",
    test: "tests/steering/score.sh::a_text_and_a_text_file_are_refused",
};

pub static INPUT_FILE_MISSING: Cause = Cause {
    id: "input-file-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading the text to score from {path}",
    cause: "no file is there: nothing at {path} is the text file --input-file names",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name the file that holds the text, or give the text itself with --input; `ls -ln {parent}` shows what that directory holds",
    test: "tests/steering/score.sh::a_missing_text_file_is_refused",
};

pub static TEXT_BLANK: Cause = Cause {
    id: "text-blank",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the text to score from {source}",
    cause: "the text is empty; give the text to score: it holds no character but whitespace, so there is no token to read",
    observed: "{source}: {bytes} bytes, all whitespace",
    remedy: "give a text that holds at least one word",
    test: "tests/steering/score.sh::an_empty_text_is_refused",
};

pub static LAYER_UNCARRIED: Cause = Cause {
    id: "layer-uncarried",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: SCORE_STEP,
    cause: "the artifact carries no vector at layer {layer}; it carries layers {carried}, and a text is scored along the direction at the layer it is read at",
    observed: "--layer {layer}; the {trait} artifact carries layers {carried}",
    remedy: "give --layer one of {carried}, or train a vector at layer {layer} with ster vector train --layers {layer}",
    test: "tests/steering/score.sh::a_layer_the_artifact_lacks_is_refused",
};

pub static DIRECTION_LENGTHLESS: Cause = Cause {
    id: "direction-lengthless",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: SCORE_STEP,
    cause: "the artifact's direction at layer {layer} has no usable length ({length}): every value is zero, or too small to measure, so no token has a length along it",
    observed: "the {trait} artifact: layer {layer}, {width} values, length {length}",
    remedy: "train the vector again with ster vector train, or score at another of its layers ({carried})",
    test: "tests/steering/score.sh::a_direction_without_length_is_refused",
};

/// `ster vector score`: a text's tokens along an artifact's direction.
pub static SCORE: Operation = Operation {
    id: "ster.vector.score",
    entries: &["ster vector score", "ster request vector/score"],
    uses: &[&FILE, &ARTIFACT, &LOAD, &TEMPLATE, &STEER, &MODEL_LAYERS, &ENCODE],
    summary: "read a text once at one layer and project each token's state onto the artifact's unit direction there",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &INPUT_ABSENT,
        &INPUT_TWICE,
        &INPUT_FILE_MISSING,
        &TEXT_BLANK,
        &LAYER_UNCARRIED,
        &DIRECTION_LENGTHLESS,
    ],
};
