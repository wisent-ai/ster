# How a steering artifact fails

Every declared way reading, checking and writing a steering artifact fails:
the JSON document `ster vector train` and `ster tune bipo` write, one vector
per layer it steers. `ster vector inspect` reads one; every command that
takes `--vector` reads one the same way, and steering a model with it is on
[runtime.md](runtime.md). How a refusal reads and which status it exits with
is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/steering.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.steering.artifact`

read a steering artifact's JSON, check every vector is one layer's finite direction as wide as the artifact says, and write one

Started through: `ster vector inspect`, `ster request vector/inspect`. Uses: `ster.file`. Failure catalogue `ster.steering.artifact`: 7 declared cause(s) of its own.

<a id="ster-steering-artifact-artifact-missing"></a>

### `ster.steering.artifact.artifact-missing`

- **What failed:** reading steering artifact {path}
- **Cause:** no file is there: nothing at {path} is the steering artifact named
- **Observed state shown:** path: {path} ({state}); its directory {parent}; the system answered: {error}
- **Fix:** name the JSON file ster vector train or ster tune bipo wrote; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/artifact.sh::a_missing_artifact_is_refused`

<a id="ster-steering-artifact-adapter-given"></a>

### `ster.steering.artifact.adapter-given`

- **What failed:** reading steering artifact {path}
- **Cause:** {path} is a LoRA adapter sidecar, not a steering artifact: it carries rank and targets where a steering artifact carries trait_name and vectors
- **Observed state shown:** {path}: an adapter sidecar
- **Fix:** pass a steering artifact with --vector and a LoRA adapter with --adapter
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/steering/artifact.sh::an_adapter_sidecar_given_as_a_vector_is_refused`

<a id="ster-steering-artifact-schema-unsupported"></a>

### `ster.steering.artifact.schema-unsupported`

- **What failed:** checking the {trait_name} steering artifact
- **Cause:** artifact schema {schema} is unsupported; this Ster build reads schema {supported}
- **Observed state shown:** schema_version: {schema}
- **Fix:** read it with the Ster release that wrote it, or train the vector again with this one
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/artifact.sh::an_artifact_of_another_schema_is_refused`

<a id="ster-steering-artifact-product-foreign"></a>

### `ster.steering.artifact.product-foreign`

- **What failed:** checking the {trait_name} steering artifact
- **Cause:** the artifact belongs to product {product}, not Ster
- **Observed state shown:** product: {product}
- **Fix:** read it with the product that wrote it; Ster reads the vectors ster vector train writes
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/artifact.sh::an_artifact_of_another_product_is_refused`

<a id="ster-steering-artifact-vectors-none"></a>

### `ster.steering.artifact.vectors-none`

- **What failed:** checking the {trait_name} steering artifact
- **Cause:** the artifact holds no steering vector to add
- **Observed state shown:** hidden_size: {hidden_size}; vectors: {count}
- **Fix:** train the vector again with ster vector train; an artifact carries one vector per layer it steers
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/artifact.sh::an_artifact_without_vectors_is_refused`

<a id="ster-steering-artifact-vector-width"></a>

### `ster.steering.artifact.vector-width`

- **What failed:** checking the {trait_name} steering artifact
- **Cause:** the vector at layer {layer} holds {width} values where the artifact's hidden_size is {hidden_size}
- **Observed state shown:** layer {layer}: {width} values
- **Fix:** use the artifact ster vector train wrote, unedited
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/artifact.sh::a_vector_of_another_width_is_refused`

<a id="ster-steering-artifact-vector-not-finite"></a>

### `ster.steering.artifact.vector-not-finite`

- **What failed:** checking the {trait_name} steering artifact
- **Cause:** the vector at layer {layer} holds a value that is not finite, which would carry NaN through every steered forward pass
- **Observed state shown:** layer {layer}
- **Fix:** train the vector again; a run that diverged writes values that are not finite
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/steering/artifact.sh::a_vector_that_is_not_finite_is_refused`
