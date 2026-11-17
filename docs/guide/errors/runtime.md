# How a loaded model's runtime fails

Every declared way a model fails once its weights are mapped: turning a
prompt, a training example, a preference side, a conversation or an answer
label into the token ids the checkpoint expects, through its tokenizer and,
when the run applies it, its chat template. Loading the model is on
[model.md](model.md). How a refusal reads and which status it exits with is
on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/runtime.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.runtime.device`

read --device and --precision and open the device the weights are mapped on, at the dtype it can multiply in

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.runtime.device`: 6 declared cause(s) of its own.

<a id="ster-runtime-device-device-unknown"></a>

### `ster.runtime.device.device-unknown`

- **What failed:** reading the device
- **Cause:** unknown device {value}; expected cpu, metal, cuda, or cuda:N
- **Observed state shown:** --device {value}
- **Fix:** give --device cpu, metal, cuda (the first GPU) or cuda:N (the GPU at N, counted from 0)
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/device.sh::an_unknown_device_is_refused`

<a id="ster-runtime-device-ordinal-invalid"></a>

### `ster.runtime.device.ordinal-invalid`

- **What failed:** reading the device
- **Cause:** --device {value} names no CUDA device: the ordinal after cuda: is {ordinal}, which is no whole number counted from 0
- **Observed state shown:** --device {value}; reading {ordinal} as a number: {error}
- **Fix:** give --device cuda:N with N the GPU's index as the driver counts them (nvidia-smi lists them), such as cuda:1, or cuda for the first
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/device.sh::a_cuda_ordinal_that_is_no_number_is_refused`

<a id="ster-runtime-device-device-unbuilt"></a>

### `ster.runtime.device.device-unbuilt`

- **What failed:** opening the {device} device
- **Cause:** this Ster binary was built without the {device} feature
- **Observed state shown:** --device {device}
- **Fix:** run with --device cpu, or install the Ster build for this machine that carries {device}
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/device.sh::a_device_this_build_lacks_is_refused`

<a id="ster-runtime-device-device-unopened"></a>

### `ster.runtime.device.device-unopened`

- **What failed:** opening the {device} device
- **Cause:** this build carries the {device} backend, and the machine's {device} driver opened no device: no such GPU is present, or its driver refused (too old, or held by another process)
- **Observed state shown:** --device {device}; the driver answered: {error}
- **Fix:** run with --device cpu, or check that this machine's {device} driver answers
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/device.sh::a_device_that_does_not_open_is_refused`

<a id="ster-runtime-device-precision-unknown"></a>

### `ster.runtime.device.precision-unknown`

- **What failed:** reading the precision
- **Cause:** unknown precision {value}; expected f32, f16, or bf16
- **Observed state shown:** --precision {value}
- **Fix:** give --precision f32, f16 or bf16
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/device.sh::an_unknown_precision_is_refused`

<a id="ster-runtime-device-bf16-on-cpu"></a>

### `ster.runtime.device.bf16-on-cpu`

- **What failed:** mapping the weights of {model} at bf16 on the cpu
- **Cause:** bf16 has no CPU matmul kernel in this Candle build, so the run would load the whole checkpoint and then fail at its first projection
- **Observed state shown:** --precision bf16 --device cpu (cpu is the device when --device is left out); nothing of {model} was read
- **Fix:** use --precision f16 for half precision on cpu, or --device metal or --device cuda for bf16
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/device.sh::bf16_on_the_cpu_is_refused`

## `ster.runtime.encode`

tokenize a prompt, a training example, a preference side, a conversation or an answer label the way this checkpoint expects, through its chat template when the run applies one

Started through: run only by the functionalities that use it. Uses: `ster.chat.template`. Failure catalogue `ster.runtime.encode`: 7 declared cause(s) of its own.

<a id="ster-runtime-encode-text-blank"></a>

### `ster.runtime.encode.text-blank`

- **What failed:** encoding the {what}
- **Cause:** the {what} is empty or only whitespace, so there is nothing to encode
- **Observed state shown:** {what}: {text}
- **Fix:** give the {what} at least one word
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/encode.sh::a_blank_prompt_is_refused`

<a id="ster-runtime-encode-tokens-none"></a>

### `ster.runtime.encode.tokens-none`

- **What failed:** encoding the {what}
- **Cause:** this model's tokenizer turns {text} into no tokens, so the model has nothing to read
- **Observed state shown:** {what}: {text}; token ids: []
- **Fix:** give the {what} text this model's tokenizer keeps, or use the checkpoint's own tokenizer.json if a normalizer added to it removes this text
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/encode.sh::text_the_tokenizer_drops_is_refused`

<a id="ster-runtime-encode-tokenizer-failed"></a>

### `ster.runtime.encode.tokenizer-failed`

- **What failed:** encoding the {what}
- **Cause:** the tokenizers library refused to encode the text: the tokenizer.json's normalizer, pre-tokenizer or model raised on it (a byte its vocabulary cannot represent, a pattern that fails on the text)
- **Observed state shown:** {what}: {text}; the library answered: {error}
- **Fix:** use the tokenizer.json the checkpoint was published with, unedited
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/encode.sh::a_text_the_tokenizer_raises_on_is_refused`

<a id="ster-runtime-encode-piece-unknown"></a>

### `ster.runtime.encode.piece-unknown`

- **What failed:** encoding the {what}
- **Cause:** the text holds a piece the tokenizer's vocabulary lacks, and its tokenizer.json names no unknown token (unk_id) to stand in for it
- **Observed state shown:** {what}: {text}; the tokenizer holds {vocabulary} tokens; the library answered: {error}
- **Fix:** write the {what} with words the vocabulary holds, or use a tokenizer.json whose model names an unk_token
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/encode.sh::a_text_the_tokenizer_refuses_is_refused`

<a id="ster-runtime-encode-system-unplaceable"></a>

### `ster.runtime.encode.system-unplaceable`

- **What failed:** encoding a training example
- **Cause:** the example has a system prompt, which only a chat template can place, and this run's chat template is {status}
- **Observed state shown:** system: {system}; chat template: {status}
- **Fix:** train a model whose tokenizer_config.json carries a chat_template with --chat-template auto, or remove the example's system key
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/tuning/runs/examples.sh::a_system_prompt_without_a_chat_template_is_refused`

<a id="ster-runtime-encode-conversation-untemplated"></a>

### `ster.runtime.encode.conversation-untemplated`

- **What failed:** encoding a conversation of several turns
- **Cause:** only the model's chat template knows how turns are delimited, and this run's chat template is {status}
- **Observed state shown:** chat template: {status}
- **Fix:** converse with a model whose tokenizer_config.json carries a chat_template, with --chat-template auto
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/encode.sh::a_conversation_without_a_chat_template_is_refused`

<a id="ster-runtime-encode-label-unspelled"></a>

### `ster.runtime.encode.label-unspelled`

- **What failed:** finding the token of answer label {label}
- **Cause:** this tokenizer has no single token for {label}, so the model's answer cannot be read from one position
- **Observed state shown:** spellings looked up: {spellings}
- **Fix:** use a model whose vocabulary holds each answer letter as one token; every Llama, Qwen and Mistral vocabulary does
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/encode.sh::a_label_the_vocabulary_lacks_is_refused`

## `ster.runtime.layers`

check that every layer an artifact or a read names is one of the model's

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.runtime.layers`: 1 declared cause(s) of its own.

<a id="ster-runtime-layers-layer-outside"></a>

### `ster.runtime.layers.layer-outside`

- **What failed:** reading layer {layer} of the model
- **Cause:** layer {layer} is outside the model's {count} layers
- **Observed state shown:** layers named: {layers}; the model has layers 0 to {last}
- **Fix:** name a layer from 0 to {last}, or use the artifact with the model it was trained on
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/steer.sh::a_vector_past_the_model_layers_is_refused`

## `ster.runtime.steer`

check that each steering artifact was trained for this model at its width and layers, and scale every vector by its strength

Started through: run only by the functionalities that use it. Uses: `ster.runtime.layers`. Failure catalogue `ster.runtime.steer`: 4 declared cause(s) of its own.

<a id="ster-runtime-steer-model-other"></a>

### `ster.runtime.steer.model-other`

- **What failed:** steering {model} with the {trait_name} artifact
- **Cause:** the artifact was trained for model {trained}, current model is {model}
- **Observed state shown:** trained for: {trained}
- **Fix:** steer the model the artifact was trained for, --model {trained}, or train a vector for this one with ster vector train
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/steer.sh::a_vector_for_another_model_is_refused`

<a id="ster-runtime-steer-width-other"></a>

### `ster.runtime.steer.width-other`

- **What failed:** steering {model} with the {trait_name} artifact
- **Cause:** artifact width {trained} does not match model width {width}
- **Observed state shown:** artifact hidden_size: {trained}; model hidden_size: {width}
- **Fix:** steer the model the artifact was trained for; a vector fits only the width it was trained at
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/steer.sh::a_vector_of_another_width_is_refused`

<a id="ster-runtime-steer-strength-not-finite"></a>

### `ster.runtime.steer.strength-not-finite`

- **What failed:** scaling the steering vectors
- **Cause:** steering strength {strength} is not finite
- **Observed state shown:** --strength {strengths}
- **Fix:** give every --strength a finite number, such as 4
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/steer.sh::a_strength_that_is_not_finite_is_refused`

<a id="ster-runtime-steer-strengths-zero"></a>

### `ster.runtime.steer.strengths-zero`

- **What failed:** scaling the steering vectors
- **Cause:** every steering strength is zero, so the vectors would steer nothing
- **Observed state shown:** --strength {strengths}
- **Fix:** give at least one --strength other than zero, or leave --vector out to generate unsteered
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/steer.sh::strengths_of_zero_are_refused`

## `ster.runtime.generate`

sample a continuation of an encoded prompt within the model's context, with steering artifacts added when given, and spell the sampled tokens

Started through: run only by the functionalities that use it. Uses: `ster.runtime.encode`, `ster.runtime.steer`. Failure catalogue `ster.runtime.generate`: 3 declared cause(s) of its own.

<a id="ster-runtime-generate-new-tokens-none"></a>

### `ster.runtime.generate.new-tokens-none`

- **What failed:** sampling a continuation
- **Cause:** max_new_tokens must be greater than zero, or there is nothing to sample
- **Observed state shown:** --max-new-tokens {tokens}; the prompt is {prompt} tokens of {model}
- **Fix:** give --max-new-tokens at least 1
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/generate.sh::no_new_token_is_refused`

<a id="ster-runtime-generate-prompt-too-long"></a>

### `ster.runtime.generate.prompt-too-long`

- **What failed:** sampling a continuation
- **Cause:** the prompt is {tokens} tokens, and this model's context holds fewer than {context}
- **Observed state shown:** prompt: {tokens} tokens; max_position_embeddings: {context}
- **Fix:** shorten the prompt below {context} tokens, or use a model with a longer context
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/generate.sh::a_prompt_past_the_context_is_refused`

<a id="ster-runtime-generate-unspellable"></a>

### `ster.runtime.generate.unspellable`

- **What failed:** spelling the sampled tokens {tokens}
- **Cause:** the tokenizers library cannot turn the sampled ids back into text: an id lies outside the vocabulary its tokenizer.json holds, or its decoder raised on them
- **Observed state shown:** token ids: {tokens}; the tokenizer holds {vocabulary} tokens; the library answered: {error}
- **Fix:** use the tokenizer.json the checkpoint was published with, unedited
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/generate.sh::tokens_the_tokenizer_cannot_spell_are_refused`

## `ster.runtime.vectors`

pair every --vector with the --strength in its place and read each steering artifact, before any weight is mapped

Started through: run only by the functionalities that use it. Uses: `ster.steering.artifact`. Failure catalogue `ster.runtime.vectors`: 1 declared cause(s) of its own.

<a id="ster-runtime-vectors-strengths-unpaired"></a>

### `ster.runtime.vectors.strengths-unpaired`

- **What failed:** pairing each --vector with its --strength
- **Cause:** every --vector needs its own --strength, and this run gave {vectors} vector(s) and {strengths} strength(s); Ster assumes no steering scale
- **Observed state shown:** --vector given {vectors} time(s), --strength {strengths} time(s)
- **Fix:** give one --strength after each --vector, in the same order
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/generate.sh::a_vector_without_its_strength_is_refused`

## `ster.generate`

read every steering artifact and the system turn before any weight, load the model with its adapter when one is named, and sample a continuation of each prompt

Started through: `ster generate`, `ster request generate`. Uses: `ster.runtime.vectors`, `ster.prompts`, `ster.model.load`, `ster.runtime.encode`, `ster.runtime.generate`, `ster.file`. Failure catalogue `ster.generate`: 1 declared cause(s) of its own.

<a id="ster-generate-system-empty"></a>

### `ster.generate.system-empty`

- **What failed:** reading the system turn {path}
- **Cause:** the file holds no text, so there is no system turn to give the model
- **Observed state shown:** --system {path}: {bytes} bytes of whitespace
- **Fix:** write the system turn into {path}, or leave --system out
- **Code:** `config`, exit 2; retrying unchanged will not help
- **Test:** `tests/runtime/generate.sh::an_empty_system_turn_is_refused`
