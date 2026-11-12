//! How rewriting a checkpoint with a trained adapter fails
//! (src/tune/artifact/merge.rs, export.rs, merge/checkpoint.rs): fitting
//! the adapter to the checkpoint it names, and reading and writing the
//! checkpoint directory a merge or an ablation produces.

use wisent_errors::{Cause, Code, Operation};

use super::super::file::FILE;
use super::super::model::weights::LOAD;
use super::DOCS;

pub static REWARD_UNMERGEABLE: Cause = Cause {
    id: "reward-unmergeable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "merging adapter {path}",
    cause: "the artifact is a {kind} model, not a generation adapter: a merge folds an adapter's low-rank updates into the base and writes no head, so the merged model would drop the head the reward model was trained to score with",
    observed: "kind: {kind}; model: {model}",
    remedy: "name an artifact ster tune sft, dpo, grpo or decide wrote; `ster tune inspect {path}` prints an artifact's kind",
    test: "tests/tuning/scales/rewrite.sh::a_reward_model_is_refused_by_merge",
};

pub static REWARD_UNEXPORTABLE: Cause = Cause {
    id: "reward-unexportable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "exporting adapter {path}",
    cause: "the artifact is a {kind} model, not a generation adapter: a PEFT adapter directory holds low-rank updates only and has no place for the reward model's head",
    observed: "kind: {kind}; model: {model}",
    remedy: "export an artifact ster tune sft, dpo, grpo or decide wrote; `ster tune inspect {path}` prints an artifact's kind",
    test: "tests/tuning/scales/rewrite.sh::a_reward_model_is_refused_by_export",
};

pub static MODEL_OTHER: Cause = Cause {
    id: "model-other",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step} artifact {path}",
    cause: "the artifact was trained for model {trained}, current model is {model}",
    observed: "trained for: {trained}",
    remedy: "name the model the artifact was trained for with --model {trained}",
    test: "tests/tuning/scales/rewrite.sh::an_artifact_for_another_model_is_refused",
};

pub static GGUF_UNMERGEABLE: Cause = Cause {
    id: "gguf-unmergeable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "merging artifact {path}",
    cause: "{model} is a GGUF checkpoint, and an adapter is merged into safetensors weights only: a GGUF base holds quantized tensors, and writing them back dequantized would publish a model of another size and precision under the base's name",
    observed: "{model}: a .gguf file",
    remedy: "run the adapter on it unmerged (`ster generate --model {model} --adapter {path} --prompt …`), or merge into the base model's safetensors release",
    test: "tests/tuning/scales/rewrite.sh::a_gguf_checkpoint_is_refused_by_merge",
};

pub static GGUF_UNEXPORTABLE: Cause = Cause {
    id: "gguf-unexportable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "exporting artifact {path}",
    cause: "{model} is a GGUF checkpoint, and a PEFT adapter names the modules of the base model's safetensors release, which a GGUF file's tensor names are not",
    observed: "{model}: a .gguf file",
    remedy: "export against the base model's safetensors release (`ster tune export --model <release> --adapter {path} --format peft --output <directory>`)",
    test: "tests/tuning/scales/rewrite.sh::a_gguf_checkpoint_is_refused_by_export",
};

pub static GGUF_UNABLATABLE: Cause = Cause {
    id: "gguf-unablatable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "ablating artifact {path}",
    cause: "{model} is a GGUF checkpoint, and a direction is projected out of safetensors weights only: a GGUF file holds quantized tensors Ster does not write back",
    observed: "{model}: a .gguf file",
    remedy: "ablate the base model's safetensors release (`ster vector ablate --model <release> --vector {path} --strength <share> --output <directory>`)",
    test: "tests/tuning/scales/rewrite.sh::a_gguf_checkpoint_is_refused_by_ablate",
};

pub static WIDTH_OTHER: Cause = Cause {
    id: "width-other",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step} artifact {path}",
    cause: "artifact width {trained} does not match model width {width}",
    observed: "hidden_size: {width}",
    remedy: "name the model the artifact was trained for; an artifact fits only the width it was trained at",
    test: "tests/tuning/scales/rewrite.sh::an_artifact_of_another_width_is_refused",
};

pub static LAYER_PAST: Cause = Cause {
    id: "layer-past",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step} artifact {path}",
    cause: "layer {layer} is outside the model's {layers} layers",
    observed: "artifact layers: {artifact_layers}",
    remedy: "name the model the artifact was trained for; its layers run from 0 to {last}",
    test: "tests/tuning/scales/rewrite.sh::an_artifact_past_the_model_layers_is_refused",
};

pub static PROJECTION_ABSENT: Cause = Cause {
    id: "projection-absent",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step} the {target} projection of layer {layer}",
    cause: "this model has no {target} projection at layer {layer}",
    observed: "artifact targets: {targets}",
    remedy: "name the checkpoint the artifact was trained on",
    test: "tests/tuning/scales/rewrite.sh::a_layer_without_the_projection_is_refused",
};

pub static FUSED_UNEXPORTABLE: Cause = Cause {
    id: "fused-unexportable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "exporting the {target} projection of layer {layer}",
    cause: "this model stores the {target} projection fused with others in one tensor, and PEFT has no module of its own for one part of a fused tensor",
    observed: "tensor: {tensor}",
    remedy: "merge the adapter with `ster tune merge --model {model} --adapter {path}` instead, which writes the update into the rows the projection owns",
    test: "tests/tuning/scales/rewrite.sh::a_fused_projection_is_refused_for_export",
};

pub static TRANSPOSED_UNEXPORTABLE: Cause = Cause {
    id: "transposed-unexportable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "exporting the {target} projection of layer {layer}",
    cause: "this model stores the {target} projection transposed (GPT-2's Conv1D), and a PEFT LoRA module's factors are laid out for a Linear weight, so PEFT would apply them to the wrong axes",
    observed: "tensor: {tensor}",
    remedy: "merge the adapter with `ster tune merge --model {model} --adapter {path}` instead",
    test: "tests/tuning/scales/rewrite.sh::a_transposed_projection_is_refused_for_export",
};

pub static FUSED_UNABLATABLE: Cause = Cause {
    id: "fused-unablatable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "ablating the {target} projection of layer {layer}",
    cause: "this model stores the {target} projection fused with others in one tensor, and an ablation rewrites each projection's whole tensor: Ster does not rewrite one part of a fused tensor",
    observed: "tensor: {tensor}",
    remedy: "ablate a checkpoint of a family that stores each projection as its own tensor, or steer at run time with `ster generate --model {model} --vector {path}`",
    test: "tests/tuning/scales/rewrite.sh::a_fused_projection_is_refused_for_ablation",
};

pub static TENSOR_MISSING: Cause = Cause {
    id: "tensor-missing",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step} the {target} projection of layer {layer}",
    cause: "{holder} holds no tensor {name}",
    observed: "{holder}: {count} tensors",
    remedy: "name the checkpoint and the artifact that belong together: `ster tune inspect` prints the artifact's tensors",
    test: "tests/tuning/scales/rewrite.sh::a_checkpoint_without_the_projection_is_refused",
};

pub static UPDATE_UNFIT: Cause = Cause {
    id: "update-unfit",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step} the {target} projection of layer {layer}",
    cause: "the update the artifact carries has another shape than {name} as the checkpoint stores it: the artifact was trained on another model or width",
    observed: "{name}: {shape} in {path}; Candle answered: {error}",
    remedy: "name the checkpoint the artifact was trained on; `ster tune inspect` prints the model and the shapes the artifact holds",
    test: "tests/tuning/scales/rewrite.sh::an_update_that_does_not_fit_is_refused",
};

pub static UPDATE_UNCOMPUTED: Cause = Cause {
    id: "update-uncomputed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step} the {target} projection of layer {layer}",
    cause: "the shapes fit, and the device did not compute the update into {name}: it has no memory left for the dense update, or does not compute in the weight's dtype",
    observed: "{name}: {shape} in {path}; the device answered: {error}",
    remedy: "run again with `--device cpu`, where the update is computed in host memory",
    test: "tests/tuning/scales/rewrite.sh::an_update_the_device_does_not_compute_is_refused",
};

/// Fitting a trained artifact to the checkpoint it is written into.
pub static FIT: Operation = Operation {
    id: "ster.tune.fit",
    entries: &[],
    uses: &[&LOAD],
    summary: "fit a trained artifact to the safetensors checkpoint it names: its kind, model, width, layers, and the tensor of every projection it rewrites",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &REWARD_UNMERGEABLE,
        &REWARD_UNEXPORTABLE,
        &MODEL_OTHER,
        &GGUF_UNMERGEABLE,
        &GGUF_UNEXPORTABLE,
        &GGUF_UNABLATABLE,
        &WIDTH_OTHER,
        &LAYER_PAST,
        &PROJECTION_ABSENT,
        &FUSED_UNEXPORTABLE,
        &TRANSPOSED_UNEXPORTABLE,
        &FUSED_UNABLATABLE,
        &TENSOR_MISSING,
        &UPDATE_UNFIT,
        &UPDATE_UNCOMPUTED,
    ],
};

pub static TENSOR_TWICE: Cause = Cause {
    id: "tensor-twice",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading the weights of {model}",
    cause: "model weight {name} appears in more than one weight file, so which one the checkpoint means is unknown",
    observed: "{name} in {path} and an earlier shard",
    remedy: "use a checkpoint whose shards hold each tensor once, as model.safetensors.index.json lists them",
    test: "tests/tuning/scales/rewrite.sh::a_tensor_in_two_shards_is_refused",
};

/// Reading a checkpoint's tensors and writing a rewritten checkpoint.
pub static WRITE: Operation = Operation {
    id: "ster.tune.write",
    entries: &[],
    uses: &[&FILE],
    summary: "read every tensor of a safetensors checkpoint and write the rewritten tensors beside copies of its config, tokenizer and chat template",
    docs: DOCS,
    warnings: &[],
    causes: &[&TENSOR_TWICE],
};

/// `ster tune merge`: an adapter folded into its base checkpoint.
pub static MERGE: Operation = Operation {
    id: "ster.tune.merge",
    entries: &["ster tune merge", "ster request tune/merge"],
    uses: &[&FIT, &WRITE],
    summary: "fold a LoRA adapter into its base's safetensors weights and write a standalone checkpoint",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};

/// `ster tune export`: an adapter written as a PEFT adapter directory.
pub static EXPORT: Operation = Operation {
    id: "ster.tune.export",
    entries: &["ster tune export", "ster request tune/export"],
    uses: &[&FIT, &WRITE],
    summary: "write a LoRA adapter as a PEFT adapter directory for vLLM and peft, beside its base's tokenizer",
    docs: DOCS,
    warnings: &[],
    causes: &[],
};
