//! How a steering artifact fails (src/artifact): reading the JSON document
//! a direction travels in, checking its vectors, and writing one. A file
//! that cannot be read or written, or is not JSON of the artifact's form, is
//! the `ster.file` layer's cause.

use wisent_errors::{Cause, Code, Operation};

use super::file;

/// What a steering artifact is, for the `ster.file` refusals that name it.
pub const FORM: &str = "the JSON `ster vector train` or `ster tune bipo` writes";

/// docs/guide/errors/steering.md, where these functionalities are listed.
const DOCS: &str = page!("steering");

pub static MISSING: Cause = Cause {
    id: "artifact-missing",
    code: Code::NotFound,
    usage: false,
    exit: None,
    failed: "reading steering artifact {path}",
    cause: "no file is there: nothing at {path} is the steering artifact named",
    observed: "path: {path} ({state}); its directory {parent}; the system answered: {error}",
    remedy: "name the JSON file ster vector train or ster tune bipo wrote; `ls -ln {parent}` shows what that directory holds",
    test: "tests/steering/artifact.sh::a_missing_artifact_is_refused",
};

pub static ADAPTER_GIVEN: Cause = Cause {
    id: "adapter-given",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading steering artifact {path}",
    cause: "{path} is a LoRA adapter sidecar, not a steering artifact: it carries rank and targets where a steering artifact carries trait_name and vectors",
    observed: "{path}: an adapter sidecar",
    remedy: "pass a steering artifact with --vector and a LoRA adapter with --adapter",
    test: "tests/steering/artifact.sh::an_adapter_sidecar_given_as_a_vector_is_refused",
};

pub static SCHEMA_UNSUPPORTED: Cause = Cause {
    id: "schema-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the {trait_name} steering artifact",
    cause: "artifact schema {schema} is unsupported; this Ster build reads schema {supported}",
    observed: "schema_version: {schema}",
    remedy: "read it with the Ster release that wrote it, or train the vector again with this one",
    test: "tests/steering/artifact.sh::an_artifact_of_another_schema_is_refused",
};

pub static PRODUCT_FOREIGN: Cause = Cause {
    id: "product-foreign",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the {trait_name} steering artifact",
    cause: "the artifact belongs to product {product}, not Ster",
    observed: "product: {product}",
    remedy: "read it with the product that wrote it; Ster reads the vectors ster vector train writes",
    test: "tests/steering/artifact.sh::an_artifact_of_another_product_is_refused",
};

pub static VECTORS_NONE: Cause = Cause {
    id: "vectors-none",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the {trait_name} steering artifact",
    cause: "the artifact holds no steering vector to add",
    observed: "hidden_size: {hidden_size}; vectors: {count}",
    remedy: "train the vector again with ster vector train; an artifact carries one vector per layer it steers",
    test: "tests/steering/artifact.sh::an_artifact_without_vectors_is_refused",
};

pub static VECTOR_WIDTH: Cause = Cause {
    id: "vector-width",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the {trait_name} steering artifact",
    cause: "the vector at layer {layer} holds {width} values where the artifact's hidden_size is {hidden_size}",
    observed: "layer {layer}: {width} values",
    remedy: "use the artifact ster vector train wrote, unedited",
    test: "tests/steering/artifact.sh::a_vector_of_another_width_is_refused",
};

pub static VECTOR_NOT_FINITE: Cause = Cause {
    id: "vector-not-finite",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "checking the {trait_name} steering artifact",
    cause: "the vector at layer {layer} holds a value that is not finite, which would carry NaN through every steered forward pass",
    observed: "layer {layer}",
    remedy: "train the vector again; a run that diverged writes values that are not finite",
    test: "tests/steering/artifact.sh::a_vector_that_is_not_finite_is_refused",
};

/// Reading, checking and writing one steering artifact.
pub static ARTIFACT: Operation = Operation {
    id: "ster.steering.artifact",
    entries: &["ster vector inspect", "ster request vector/inspect"],
    uses: &[&file::FILE],
    summary: "read a steering artifact's JSON, check every vector is one layer's finite direction as wide as the artifact says, and write one",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &MISSING,
        &ADAPTER_GIVEN,
        &SCHEMA_UNSUPPORTED,
        &PRODUCT_FOREIGN,
        &VECTORS_NONE,
        &VECTOR_WIDTH,
        &VECTOR_NOT_FINITE,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&ARTIFACT];
