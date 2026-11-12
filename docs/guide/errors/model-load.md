# How loading a model with adapters fails

Every declared way loading a model fails once its checkpoint, config and
weights are read: placing fresh LoRA adapters on a family that cannot hold
them, and attaching a trained artifact made for another model, width or
purpose. Finding the checkpoint and mapping its weights are on
[model.md](model.md), reading its config on [model-config.md](model-config.md).
How a refusal reads and which status it exits with is on
[How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/model-load.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.model.load`

load a local or Hugging Face checkpoint into Ster's runtime, place fresh LoRA adapters on it or attach a trained LoRA or reward artifact when one is named, refusing adapters its family cannot hold and an artifact trained for another model, width or purpose

Started through: run only by the functionalities that use it. Uses: `ster.model.checkpoint`, `ster.model.config`, `ster.model.weights`. Failure catalogue `ster.model.load`: 5 declared cause(s) of its own.

<a id="ster-model-load-artifact-kind"></a>

### `ster.model.load.artifact-kind`

- **What failed:** attaching artifact {path} to {model}
- **Cause:** adapter artifact is a {found}, not a {wanted}
- **Observed state shown:** kind: {kind}
- **Fix:** attach a {wanted} artifact here: `ster tune inspect {path}` prints an artifact's kind
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/adapter.sh::a_reward_model_given_as_a_generation_adapter_is_refused`

<a id="ster-model-load-artifact-model"></a>

### `ster.model.load.artifact-model`

- **What failed:** attaching artifact {path} to {model}
- **Cause:** adapter was trained for model {trained}, current model is {model}
- **Observed state shown:** trained for: {trained}
- **Fix:** load the model the artifact was trained for with --model {trained}, or train an artifact for {model}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/adapter.sh::an_adapter_for_another_model_is_refused`

<a id="ster-model-load-artifact-width"></a>

### `ster.model.load.artifact-width`

- **What failed:** attaching artifact {path} to {model}
- **Cause:** adapter width {trained} does not match model width {width}
- **Observed state shown:** hidden_size: {width}
- **Fix:** train an artifact for {model}; an artifact attaches only to the width it was trained at
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/adapter.sh::an_adapter_of_another_width_is_refused`

<a id="ster-model-load-adapters-unsupported"></a>

### `ster.model.load.adapters-unsupported`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** {detail}
- **Observed state shown:** the family's {feature}
- **Fix:** steer this model with a steering vector (`ster extract`), or train adapters on a checkpoint of a family with attention projections on every layer
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_model_that_trains_no_adapters_is_refused`

<a id="ster-model-load-target-absent"></a>

### `ster.model.load.target-absent`

- **What failed:** placing LoRA adapters on {model}
- **Cause:** this model has no single {target} projection to adapt because {why}
- **Observed state shown:** targets this model has: {choices}
- **Fix:** choose adapter targets among {choices} with --targets
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/models/loading/targets.sh::a_target_the_model_lacks_is_refused`
