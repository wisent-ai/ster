# How a computation on tensors fails

Every declared way a computation Ster runs through Candle fails, whichever
command runs it: a forward pass, a training step's backward pass and update,
and a tensor read or written. Candle hands back one error type; which of its
variants it is, never its words, decides the cause: the accelerator refusing
the operation (out of memory, or a driver failing the kernel, exit 69), a
device with no kernel for the dtype the weights are mapped at, tensors Ster
built that do not fit each other or a view a kernel does not take (defects
of Ster), a gradient through an inference kernel, and a tensor file Candle
cannot read or write. The step a refusal names is what the site was
computing, or the steps the error's context names when it reached the end
of the command undeclared. How a refusal reads and which status it exits with
is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/tensor.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
## `ster.tensor`

compute on tensors through Candle: every forward pass, training step and tensor read or write a command runs

Started through: run only by the functionalities that use it. Uses: none. Failure catalogue `ster.tensor`: 8 declared cause(s) of its own.

<a id="ster-tensor-device-refused"></a>

### `ster.tensor.device-refused`

- **What failed:** {step}
- **Cause:** the accelerator's backend (Metal or CUDA) refused the operation: it has no free memory left for the tensors, or its driver failed the kernel
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** run again with fewer rows at once (a lower --batch-size or --max-sequence) or after other work on the device ends, or with --device cpu, where host memory holds the tensors
- **Code:** `infra_down`, exit 69; retrying unchanged can help
- **Test:** `tests/runtime/tensor.sh::an_accelerator_out_of_memory_is_refused`

<a id="ster-tensor-backend-unbuilt"></a>

### `ster.tensor.backend-unbuilt`

- **What failed:** {step}
- **Cause:** the tensors live on a device whose backend this build of Ster carries no kernels for
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** run again with --device cpu, or install the Ster release built with the backend this machine's accelerator needs
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/tensor.sh::a_backend_this_build_lacks_is_refused`

<a id="ster-tensor-dtype-unsupported"></a>

### `ster.tensor.dtype-unsupported`

- **What failed:** {step}
- **Cause:** the device has no kernel for this operation at the dtype the weights are mapped at
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** run again with --precision f32, which every kernel takes, or on a device whose kernels take the precision asked for
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/tensor.sh::an_operation_without_a_kernel_for_its_dtype_is_refused`

<a id="ster-tensor-tensors-misbuilt"></a>

### `ster.tensor.tensors-misbuilt`

- **What failed:** {step}
- **Cause:** two tensors Ster combined do not fit each other (their shapes, dims, dtypes or devices differ, or an index falls outside one): Ster built them from the model's config and the input, and checks both before computing
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the model: Ster built these tensors, so this is a defect of Ster, not of the input
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/tensor.sh::tensors_that_do_not_fit_are_refused`

<a id="ster-tensor-layout-unsupported"></a>

### `ster.tensor.layout-unsupported`

- **What failed:** {step}
- **Cause:** the operation needs its input laid out contiguously or with strides the kernel takes, and Ster handed it a view that is not
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the model: Ster built these tensors, so this is a defect of Ster, not of the input
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/tensor.sh::a_view_the_kernel_does_not_take_is_refused`

<a id="ster-tensor-backward-unsupported"></a>

### `ster.tensor.backward-unsupported`

- **What failed:** {step}
- **Cause:** the training step's backward pass reached an operation Candle computes no gradient for: a differentiable pass ran a fused inference kernel
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the model: Ster built these tensors, so this is a defect of Ster, not of the input
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/tensor.sh::a_gradient_through_an_inference_kernel_is_refused`

<a id="ster-tensor-tensor-storage-failed"></a>

### `ster.tensor.tensor-storage-failed`

- **What failed:** {step}
- **Cause:** reading or writing a tensor file's bytes failed: the file is not the safetensors, npy or zip form Candle reads, or the system refused the read or write
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** check the file the step names with `ls -ln` and that it was written whole; write it again with the ster command that writes it
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/tensor.sh::a_tensor_file_candle_does_not_read_is_refused`

<a id="ster-tensor-tensor-failed-untold"></a>

### `ster.tensor.tensor-failed-untold`

- **What failed:** {step}
- **Cause:** Candle refused the operation with an error of a variant Ster does not tell apart (a message or a wrapped error of its own), so the reason is only Candle's
- **Observed state shown:** Candle's {variant} error: {error}; {within}
- **Fix:** report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the model: Ster built these tensors, so this is a defect of Ster, not of the input
- **Code:** `config`, exit 1; retrying unchanged will not help
- **Test:** `tests/runtime/tensor.sh::an_error_candle_raises_as_a_message_is_refused`
