//! How mapping a checkpoint's weights into its decoder fails
//! (src/runtime/load/base.rs, gguf/, hy_v4/, and the decoder built from
//! them), and how attaching a trained artifact at load fails
//! (src/runtime/load/mod.rs).

use wisent_errors::{Cause, Code, Operation};

use super::super::lora::artifact::ARTIFACT;
use super::super::runtime::{device::DEVICE, steer::LAYERS};
use super::DOCS;
use super::checkpoint::CHECKPOINT;
use super::decoder::DECODER;
use super::tokenizer::TOKENIZER;
use super::config::CONFIG;

pub mod adapters;

use adapters::ADAPTERS;

/// docs/guide/errors/model-load.md, where loading with adapters is listed.
const LOAD_DOCS: &str = page!("model-load");

pub static GGUF_SEVERAL: Cause = Cause {
    id: "gguf-several",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "mapping the GGUF weights of the checkpoint",
    cause: "a GGUF checkpoint is one .gguf file; this one lists {count}",
    observed: "files: {files}",
    remedy: "keep the one .gguf file the checkpoint is beside its config and tokenizer, and move the others out",
    test: "tests/models/load.sh::two_gguf_files_are_refused",
};

pub static GGUF_UNREADABLE: Cause = Cause {
    id: "gguf-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading GGUF weights {path}",
    cause: "{path} is not a GGUF file: its header does not open with the GGUF magic, or names a version or value type Candle does not read",
    observed: "path: {path} ({state}); the reader answered: {error}",
    remedy: "replace it with the complete .gguf file its publisher wrote, or move it out of the checkpoint directory",
    test: "tests/models/load.sh::a_file_that_is_not_gguf_is_refused",
};

pub static GGUF_TRUNCATED: Cause = Cause {
    id: "gguf-truncated",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading GGUF weights {path}",
    cause: "{path} ends before its header does: it was cut short, as a download that stopped leaves it",
    observed: "path: {path} ({state}); the reader answered: {error}",
    remedy: "fetch the complete .gguf file again from where it was published",
    test: "tests/models/load.sh::a_gguf_file_cut_short_is_refused",
};

pub static GGUF_TENSOR_UNREADABLE: Cause = Cause {
    id: "gguf-tensor-unreadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading tensor {tensor} of GGUF weights {path}",
    cause: "the file lists tensor {tensor} and its bytes cannot be read: the file ends before the tensor's offset, or stores it in a quantization Candle does not dequantize",
    observed: "path: {path} ({state}); asked for as {name}; the reader answered: {error}",
    remedy: "fetch the complete .gguf file again; a file whose quantization Candle does not read is published in another one beside it",
    test: "tests/models/load.sh::a_gguf_tensor_that_cannot_be_read_is_refused",
};

pub static SAFETENSORS_UNMAPPABLE: Cause = Cause {
    id: "safetensors-unmappable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "mapping {count} safetensors weight file(s)",
    cause: "the system did not map the weight files into memory: the address space has no room for them, or the system refused the mapping",
    observed: "files: {files}; Candle answered: {error}",
    remedy: "close what holds the machine's memory, or load a smaller checkpoint; `vm_stat` (macOS) or `free -h` (Linux) shows what is left",
    test: "tests/models/load.sh::weight_files_the_system_does_not_map_are_refused",
};

pub static WEIGHTS_UNFIT: Cause = Cause {
    id: "weights-unfit",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "building the decoder of {model} from its weights",
    cause: "a tensor the checkpoint stores has another shape than the decoder its config describes needs",
    observed: "{shapes}",
    remedy: "use the config the weights were written with: a tensor's shape follows from hidden_size, intermediate_size, the head counts and vocab_size",
    test: "tests/models/load.sh::weights_that_do_not_fit_the_config_are_refused",
};

pub static TENSOR_UNSTORED: Cause = Cause {
    id: "tensor-unstored",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "building the decoder of {model} from its weights",
    cause: "the decoder its config describes needs a tensor the checkpoint does not store: the config names another architecture or layer count than the weights were written for, or a shard is missing",
    observed: "{shapes}",
    remedy: "use the config the weights were written with, and every shard model.safetensors.index.json lists",
    test: "tests/models/load.sh::weights_without_a_tensor_the_config_needs_are_refused",
};

pub static WEIGHTS_UNBUILT: Cause = Cause {
    id: "weights-unbuilt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "building the decoder of {model} from its weights",
    cause: "the shapes fit, and the device did not build the decoder from them: it has no memory left for the weights, or does not compute in their dtype",
    observed: "{shapes}; Candle answered: {error}",
    remedy: "load at a smaller --precision, on `--device cpu`, or a smaller checkpoint",
    test: "tests/models/load.sh::weights_the_device_does_not_take_are_refused",
};

pub static ARTIFACT_KIND: Cause = Cause {
    id: "artifact-kind",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "attaching artifact {path} to {model}",
    cause: "adapter artifact is a {found}, not a {wanted}",
    observed: "kind: {kind}",
    remedy: "attach a {wanted} artifact here: `ster tune inspect {path}` prints an artifact's kind",
    test: "tests/models/adapter.sh::a_reward_model_given_as_a_generation_adapter_is_refused",
};

pub static ARTIFACT_MODEL: Cause = Cause {
    id: "artifact-model",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "attaching artifact {path} to {model}",
    cause: "adapter was trained for model {trained}, current model is {model}",
    observed: "trained for: {trained}",
    remedy: "load the model the artifact was trained for with --model {trained}, or train an artifact for {model}",
    test: "tests/models/adapter.sh::an_adapter_for_another_model_is_refused",
};

pub static ARTIFACT_WIDTH: Cause = Cause {
    id: "artifact-width",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "attaching artifact {path} to {model}",
    cause: "adapter width {trained} does not match model width {width}",
    observed: "hidden_size: {width}",
    remedy: "train an artifact for {model}; an artifact attaches only to the width it was trained at",
    test: "tests/models/adapter.sh::an_adapter_of_another_width_is_refused",
};

/// Mapping the weights into the decoder, and attaching an artifact.
pub static WEIGHTS: Operation = Operation {
    id: "ster.model.weights",
    entries: &[],
    uses: &[&DECODER, &super::super::file::FILE],
    summary: "map a checkpoint's safetensors or GGUF weights read-only at the run's dtype and build its decoder from them",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &GGUF_SEVERAL,
        &GGUF_UNREADABLE,
        &GGUF_TRUNCATED,
        &GGUF_TENSOR_UNREADABLE,
        &SAFETENSORS_UNMAPPABLE,
        &WEIGHTS_UNFIT,
        &TENSOR_UNSTORED,
        &WEIGHTS_UNBUILT,
    ],
};

/// Loading a model: the checkpoint found, its config read, its weights
/// mapped, and a trained artifact attached when one is named.
pub static LOAD: Operation = Operation {
    id: "ster.model.load",
    entries: &[],
    uses: &[
        &DEVICE, &CHECKPOINT, &CONFIG, &TOKENIZER, &WEIGHTS, &ARTIFACT, &LAYERS, &ADAPTERS,
    ],
    summary: "load a local or Hugging Face checkpoint into Ster's runtime, place fresh LoRA adapters on it or attach a trained LoRA or reward artifact when one is named, refusing adapters its family cannot hold and an artifact trained for another model, width or purpose",
    docs: LOAD_DOCS,
    warnings: &[],
    causes: &[
        &ARTIFACT_KIND,
        &ARTIFACT_MODEL,
        &ARTIFACT_WIDTH,
    ],
};
