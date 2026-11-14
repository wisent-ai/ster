# How LoRA adapters fail

Every declared way the rank, alpha, targets and layers of a training run's
adapters are refused, and every way reading, checking and writing a trained
adapter's artifact fails: its safetensors weights, the JSON sidecar beside
them, and the tensors the sidecar says it holds. `ster tune inspect` reads
one; every command that takes `--adapter` reads one the same way. How a
refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/lora.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.lora.spec`

read the rank, alpha, projections and layers of the LoRA adapters a run creates; every check but the layer count is made before a model loads

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.lora.spec`: 7 declared cause(s) of its own.

<a id="ster-lora-spec-target-unknown"></a>

### `ster.lora.spec.target-unknown`

- **What failed:** reading the adapter targets
- **Cause:** unknown adapter target {target}; expected one of query, key, value, output, gate, up, down
- **Observed state shown:** --targets {targets}
- **Fix:** name targets among query, key, value, output, gate, up and down (or q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), comma-separated
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::an_unknown_target_is_refused_before_the_model_loads`

<a id="ster-lora-spec-targets-none"></a>

### `ster.lora.spec.targets-none`

- **What failed:** reading the adapter targets
- **Cause:** adapter spec names no targets; choose at least one of query, key, value, output, gate, up, down
- **Observed state shown:** --targets {targets}
- **Fix:** give --targets at least one projection, such as query,value
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::no_target_is_refused_before_the_model_loads`

<a id="ster-lora-spec-target-twice"></a>

### `ster.lora.spec.target-twice`

- **What failed:** checking the adapter spec
- **Cause:** adapter spec names target {target} twice
- **Observed state shown:** targets: {targets}
- **Fix:** name each target once
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_target_named_twice_by_a_request_is_refused`

<a id="ster-lora-spec-rank-zero"></a>

### `ster.lora.spec.rank-zero`

- **What failed:** checking the adapter spec
- **Cause:** adapter rank 0 trains nothing; choose a rank of at least 1
- **Observed state shown:** --rank {rank}
- **Fix:** give --rank at least 1, such as 8
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_rank_of_zero_is_refused_before_the_model_loads`

<a id="ster-lora-spec-alpha-unusable"></a>

### `ster.lora.spec.alpha-unusable`

- **What failed:** checking the adapter spec
- **Cause:** adapter alpha {alpha} is not a positive finite number
- **Observed state shown:** --alpha {alpha}
- **Fix:** give --alpha a finite number above zero; twice the rank is a common choice
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::an_alpha_of_zero_is_refused_before_the_model_loads`

<a id="ster-lora-spec-layer-past"></a>

### `ster.lora.spec.layer-past`

- **What failed:** placing the adapters on the model's layers
- **Cause:** adapter spec names layer {layer}, but the model has {count} layers
- **Observed state shown:** --layers {layers}
- **Fix:** give --layers indexes from 0 to {last}, or all
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_layer_past_the_model_is_refused`

<a id="ster-lora-spec-layers-none"></a>

### `ster.lora.spec.layers-none`

- **What failed:** placing the adapters on the model's layers
- **Cause:** the model's config states {count} layers, so it holds no projection an adapter can be placed on
- **Observed state shown:** num_hidden_layers: {count}
- **Fix:** train on a checkpoint whose config.json states num_hidden_layers of at least 1
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_model_without_layers_is_refused`

## `ster.lora.artifact`

read a LoRA adapter's safetensors weights and JSON sidecar, check every tensor the sidecar declares is there, shaped and finite, and write one back

Started through: `ster tune inspect`, `ster request tune/inspect`. Uses: `ster.file`. Failure catalogue `ster.lora.artifact`: 18 declared cause(s) of its own.

<a id="ster-lora-artifact-adapter-missing"></a>

### `ster.lora.artifact.adapter-missing`

- **What failed:** reading adapter {path}
- **Cause:** no file is there: nothing at {path} is the adapter named
- **Observed state shown:** path: {path} ({state}); its directory {parent}
- **Fix:** name the .safetensors file a ster tune command wrote, with its .json sidecar beside it; `ls -ln {parent}` shows what that directory holds
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_missing_adapter_is_refused`

<a id="ster-lora-artifact-sidecar-missing"></a>

### `ster.lora.artifact.sidecar-missing`

- **What failed:** reading adapter {path}
- **Cause:** adapter {path} has no sidecar at {sidecar}; the pair is written together and must travel together
- **Observed state shown:** missing: {sidecar}
- **Fix:** copy {sidecar} beside the weights from where the adapter was trained
- **Code:** `not_found`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::an_adapter_without_its_sidecar_is_refused`

<a id="ster-lora-artifact-steering-given"></a>

### `ster.lora.artifact.steering-given`

- **What failed:** reading adapter {path}
- **Cause:** {sidecar} is a steering artifact, not a LoRA adapter sidecar: it carries trait_name and vectors where an adapter sidecar carries rank and targets
- **Observed state shown:** {sidecar}: a steering artifact
- **Fix:** pass a steering artifact with --vector and a LoRA adapter with --adapter
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_steering_artifact_given_as_an_adapter_is_refused`

<a id="ster-lora-artifact-schema-unsupported"></a>

### `ster.lora.artifact.schema-unsupported`

- **What failed:** checking the adapter artifact
- **Cause:** adapter artifact schema {schema} is unsupported; this Ster build reads schema {supported}
- **Observed state shown:** schema_version: {schema}
- **Fix:** read it with the Ster release that wrote it, or train the adapter again with this one
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_sidecar_of_another_schema_is_refused`

<a id="ster-lora-artifact-product-foreign"></a>

### `ster.lora.artifact.product-foreign`

- **What failed:** checking the adapter artifact
- **Cause:** adapter artifact belongs to product {product}, not Ster
- **Observed state shown:** product: {product}
- **Fix:** read it with the product that wrote it; Ster reads the adapters ster tune writes
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_sidecar_of_another_product_is_refused`

<a id="ster-lora-artifact-rank-zero"></a>

### `ster.lora.artifact.rank-zero`

- **What failed:** checking the adapter artifact
- **Cause:** the sidecar declares rank 0, and an adapter of rank 0 stores no update: every factor pair is zero columns wide, which no Ster trainer writes
- **Observed state shown:** rank: 0
- **Fix:** use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/sidecar.sh::a_sidecar_declaring_rank_zero_is_refused`

<a id="ster-lora-artifact-alpha-invalid"></a>

### `ster.lora.artifact.alpha-invalid`

- **What failed:** checking the adapter artifact
- **Cause:** the sidecar declares an alpha that is not a positive finite number, so the update's scale, alpha over rank, is zero, negative or not a number
- **Observed state shown:** alpha: {alpha}; rank: {rank}
- **Fix:** use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/sidecar.sh::a_sidecar_declaring_alpha_zero_is_refused`

<a id="ster-lora-artifact-hidden-size-zero"></a>

### `ster.lora.artifact.hidden-size-zero`

- **What failed:** checking the adapter artifact
- **Cause:** the sidecar declares hidden size 0, and an adapter is trained on a model whose residual stream is at least one wide
- **Observed state shown:** hidden_size: 0; model: {model}
- **Fix:** use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/sidecar.sh::a_sidecar_declaring_hidden_size_zero_is_refused`

<a id="ster-lora-artifact-targets-none"></a>

### `ster.lora.artifact.targets-none`

- **What failed:** checking the adapter artifact
- **Cause:** the sidecar names no target projection, so the adapter adapts nothing
- **Observed state shown:** targets: []; layers: {layers}
- **Fix:** use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/sidecar.sh::a_sidecar_naming_no_target_is_refused`

<a id="ster-lora-artifact-layers-none"></a>

### `ster.lora.artifact.layers-none`

- **What failed:** checking the adapter artifact
- **Cause:** the sidecar names no layer, so the adapter adapts nothing
- **Observed state shown:** layers: []; targets: {targets}
- **Fix:** use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/sidecar.sh::a_sidecar_naming_no_layer_is_refused`

<a id="ster-lora-artifact-target-repeated"></a>

### `ster.lora.artifact.target-repeated`

- **What failed:** checking the adapter artifact
- **Cause:** the sidecar names target {target} twice, and a projection holds one factor pair per layer, so the two entries would name the same tensors
- **Observed state shown:** targets: {targets}
- **Fix:** use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/sidecar.sh::a_sidecar_naming_a_target_twice_is_refused`

<a id="ster-lora-artifact-layer-repeated"></a>

### `ster.lora.artifact.layer-repeated`

- **What failed:** checking the adapter artifact
- **Cause:** the sidecar names layer {layer} twice, and a layer holds one factor pair per target, so the two entries would name the same tensors
- **Observed state shown:** layers: {layers}
- **Fix:** use the sidecar the ster tune command wrote beside these weights, unedited (a sidecar is plain JSON; compare it with the one a fresh `ster tune sft` writes), or train the adapter again
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/scales/sidecar.sh::a_sidecar_naming_a_layer_twice_is_refused`

<a id="ster-lora-artifact-tensor-absent"></a>

### `ster.lora.artifact.tensor-absent`

- **What failed:** checking the adapter artifact
- **Cause:** adapter artifact is missing tensor {name}, which its sidecar's targets and layers need
- **Observed state shown:** missing: {name}
- **Fix:** use the weights and the sidecar a ster tune command wrote together
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_sidecar_naming_a_layer_the_weights_lack_is_refused`

<a id="ster-lora-artifact-tensor-shape"></a>

### `ster.lora.artifact.tensor-shape`

- **What failed:** checking the adapter artifact
- **Cause:** adapter tensor {name} has {found}, expected {expected}
- **Observed state shown:** {name}: {shape}
- **Fix:** use the weights and the sidecar a ster tune command wrote together, for the model it was trained on
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_sidecar_of_another_rank_is_refused`

<a id="ster-lora-artifact-tensor-not-finite"></a>

### `ster.lora.artifact.tensor-not-finite`

- **What failed:** checking the adapter artifact
- **Cause:** adapter tensor {name} contains a non-finite value, which would carry NaN through every forward pass
- **Observed state shown:** {name}: {count} non-finite values
- **Fix:** train the adapter again with a lower --learning-rate; a run that diverged writes infinite factors
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_non_finite_factor_is_refused`

<a id="ster-lora-artifact-head-missing"></a>

### `ster.lora.artifact.head-missing`

- **What failed:** checking the adapter artifact
- **Cause:** reward artifact is missing tensor reward.head, which is the head it exists to carry
- **Observed state shown:** kind: reward, tensors: {count}
- **Fix:** use the weights ster tune reward wrote beside this sidecar
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::a_reward_sidecar_without_its_head_is_refused`

<a id="ster-lora-artifact-head-unexpected"></a>

### `ster.lora.artifact.head-unexpected`

- **What failed:** checking the adapter artifact
- **Cause:** adapter artifact carries a reward.head tensor but declares kind adapter
- **Observed state shown:** kind: adapter, with reward.head of shape {shape} among {count} tensors; trained for {model}
- **Fix:** use the sidecar ster tune reward wrote beside these weights
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::an_adapter_sidecar_over_reward_weights_is_refused`

<a id="ster-lora-artifact-width-unfit"></a>

### `ster.lora.artifact.width-unfit`

- **What failed:** fitting the adapter to the model's projections
- **Cause:** adapter tensor {name} has shape {found}, expected {expected} for the model projection
- **Observed state shown:** {name}: {found}
- **Fix:** use the adapter with the model it was trained on
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/adapters.sh::an_adapter_of_another_projection_width_is_refused`
