//! How a computation on tensors fails: every forward pass, training step,
//! tensor read and write Ster runs through Candle, the layer the other
//! functionalities use. Candle hands back one error type; which of its
//! variants it is (never its words) decides the cause, so a device out of
//! memory, a dtype a kernel lacks and a shape Ster built wrong read
//! differently and say what fixes each.
//!
//! A site that knows its step refuses through [`computing`], or through
//! [`within`] for an error whose Candle error lies deeper in its chain; an
//! error that reaches a command's end undeclared is declared there by
//! [`settle`], with the command as the step. The context lines a chain
//! carries above the Candle error are other layers' words: a refusal shows
//! them as where it was raised, never as the step.

use candle_core::Error;
use wisent_errors::{Cause, Code, Operation};

use super::{Declared, candle, declared, refuse_from};

/// docs/guide/errors/tensor.md, where this functionality is listed.
const DOCS: &str = page!("tensor");

/// What every defect of Ster's own tensor code tells its reader to do.
const REPORT: &str = "report it at https://github.com/wisent-ai/ster/issues with the command, `ster --version` and the model: Ster built these tensors, so this is a defect of Ster, not of the input";

pub static DEVICE_REFUSED: Cause = Cause {
    id: "device-refused",
    code: Code::InfraDown,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the accelerator's backend (Metal or CUDA) refused the operation: it has no free memory left for the tensors, or its driver failed the kernel",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: "run again with fewer rows at once (a lower --batch-size or --max-sequence) or after other work on the device ends, or with --device cpu, where host memory holds the tensors",
    test: "tests/runtime/tensor.sh::an_accelerator_out_of_memory_is_refused",
};

pub static BACKEND_UNBUILT: Cause = Cause {
    id: "backend-unbuilt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the tensors live on a device whose backend this build of Ster carries no kernels for",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: "run again with --device cpu, or install the Ster release built with the backend this machine's accelerator needs",
    test: "tests/runtime/tensor.sh::a_backend_this_build_lacks_is_refused",
};

pub static DTYPE_UNSUPPORTED: Cause = Cause {
    id: "dtype-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the device has no kernel for this operation at the dtype the weights are mapped at",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: "run again with --precision f32, which every kernel takes, or on a device whose kernels take the precision asked for",
    test: "tests/runtime/tensor.sh::an_operation_without_a_kernel_for_its_dtype_is_refused",
};

pub static TENSORS_MISBUILT: Cause = Cause {
    id: "tensors-misbuilt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "two tensors Ster combined do not fit each other (their shapes, dims, dtypes or devices differ, or an index falls outside one): Ster built them from the model's config and the input, and checks both before computing",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: REPORT,
    test: "tests/runtime/tensor.sh::tensors_that_do_not_fit_are_refused",
};

pub static LAYOUT_UNSUPPORTED: Cause = Cause {
    id: "layout-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the operation needs its input laid out contiguously or with strides the kernel takes, and Ster handed it a view that is not",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: REPORT,
    test: "tests/runtime/tensor.sh::a_view_the_kernel_does_not_take_is_refused",
};

pub static BACKWARD_UNSUPPORTED: Cause = Cause {
    id: "backward-unsupported",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "the training step's backward pass reached an operation Candle computes no gradient for: a differentiable pass ran a fused inference kernel",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: REPORT,
    test: "tests/runtime/tensor.sh::a_gradient_through_an_inference_kernel_is_refused",
};

pub static STORAGE_FAILED: Cause = Cause {
    id: "tensor-storage-failed",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "reading or writing a tensor file's bytes failed: the file is not the safetensors, npy or zip form Candle reads, or the system refused the read or write",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: "check the file the step names with `ls -ln` and that it was written whole; write it again with the ster command that writes it",
    test: "tests/runtime/tensor.sh::a_tensor_file_candle_does_not_read_is_refused",
};

pub static UNTOLD: Cause = Cause {
    id: "tensor-failed-untold",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "{step}",
    cause: "Candle refused the operation with an error of a variant Ster does not tell apart (a message or a wrapped error of its own), so the reason is only Candle's",
    observed: "Candle's {variant} error: {error}; {within}",
    remedy: REPORT,
    test: "tests/runtime/tensor.sh::an_error_candle_raises_as_a_message_is_refused",
};

/// Computing on tensors through Candle.
pub static TENSOR: Operation = Operation {
    id: "ster.tensor",
    entries: &[],
    uses: &[],
    summary: "compute on tensors through Candle: every forward pass, training step and tensor read or write a command runs",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &DEVICE_REFUSED,
        &BACKEND_UNBUILT,
        &DTYPE_UNSUPPORTED,
        &TENSORS_MISBUILT,
        &LAYOUT_UNSUPPORTED,
        &BACKWARD_UNSUPPORTED,
        &STORAGE_FAILED,
        &UNTOLD,
    ],
};

pub(super) static OPERATIONS: &[&Operation] = &[&TENSOR];

/// The variant `error` is, under any backtrace, context or path Candle
/// added, as a refusal names it, and the cause that variant declares.
fn classify(error: &Error) -> (&'static str, &'static Cause) {
    match candle::root(error) {
        Error::Metal(_) => ("Metal", &DEVICE_REFUSED),
        Error::Cuda(_) => ("Cuda", &DEVICE_REFUSED),
        Error::UnsupportedDTypeForOp(..) => ("UnsupportedDTypeForOp", &DTYPE_UNSUPPORTED),
        Error::UnexpectedDType { .. } => ("UnexpectedDType", &TENSORS_MISBUILT),
        Error::DTypeMismatchBinaryOp { .. } => ("DTypeMismatchBinaryOp", &TENSORS_MISBUILT),
        Error::DeviceMismatchBinaryOp { .. } => ("DeviceMismatchBinaryOp", &TENSORS_MISBUILT),
        Error::DimOutOfRange { .. } => ("DimOutOfRange", &TENSORS_MISBUILT),
        Error::DuplicateDimIndex { .. } => ("DuplicateDimIndex", &TENSORS_MISBUILT),
        Error::UnexpectedNumberOfDims { .. } => ("UnexpectedNumberOfDims", &TENSORS_MISBUILT),
        Error::UnexpectedShape { .. } => ("UnexpectedShape", &TENSORS_MISBUILT),
        Error::ShapeMismatch { .. } => ("ShapeMismatch", &TENSORS_MISBUILT),
        Error::ShapeMismatchBinaryOp { .. } => ("ShapeMismatchBinaryOp", &TENSORS_MISBUILT),
        Error::ShapeMismatchCat { .. } => ("ShapeMismatchCat", &TENSORS_MISBUILT),
        Error::ShapeMismatchSplit { .. } => ("ShapeMismatchSplit", &TENSORS_MISBUILT),
        Error::OnlySingleDimension { .. } => ("OnlySingleDimension", &TENSORS_MISBUILT),
        Error::EmptyTensor { .. } => ("EmptyTensor", &TENSORS_MISBUILT),
        Error::NarrowInvalidArgs { .. } => ("NarrowInvalidArgs", &TENSORS_MISBUILT),
        Error::InvalidIndex { .. } => ("InvalidIndex", &TENSORS_MISBUILT),
        Error::BroadcastIncompatibleShapes { .. } => {
            ("BroadcastIncompatibleShapes", &TENSORS_MISBUILT)
        }
        Error::OpRequiresAtLeastOneTensor { .. } => ("OpRequiresAtLeastOneTensor", &TENSORS_MISBUILT),
        Error::OpRequiresAtLeastTwoTensors { .. } => {
            ("OpRequiresAtLeastTwoTensors", &TENSORS_MISBUILT)
        }
        Error::CannotSetVar { .. } => ("CannotSetVar", &TENSORS_MISBUILT),
        Error::TryFromIntError(_) => ("TryFromIntError", &TENSORS_MISBUILT),
        Error::MatMulUnexpectedStriding(_) => ("MatMulUnexpectedStriding", &LAYOUT_UNSUPPORTED),
        Error::RequiresContiguous { .. } => ("RequiresContiguous", &LAYOUT_UNSUPPORTED),
        Error::BackwardNotSupported { .. } => ("BackwardNotSupported", &BACKWARD_UNSUPPORTED),
        Error::Io(_) => ("Io", &STORAGE_FAILED),
        Error::SafeTensor(_) => ("SafeTensor", &STORAGE_FAILED),
        Error::UnsupportedSafeTensorDtype(_) => ("UnsupportedSafeTensorDtype", &STORAGE_FAILED),
        Error::Npy(_) => ("Npy", &STORAGE_FAILED),
        Error::Zip(_) => ("Zip", &STORAGE_FAILED),
        Error::CannotFindTensor { .. } => ("CannotFindTensor", &STORAGE_FAILED),
        Error::Msg(_) => ("Msg", &UNTOLD),
        Error::Wrapped(_) => ("Wrapped", &UNTOLD),
        Error::WrappedContext { .. } => ("WrappedContext", &UNTOLD),
        Error::UnwrapNone => ("UnwrapNone", &UNTOLD),
        Error::ParseInt(_) => ("ParseInt", &STORAGE_FAILED),
        Error::FromUtf8(_) => ("FromUtf8", &STORAGE_FAILED),
        Error::NotCompiledWithCudaSupport => ("NotCompiledWithCudaSupport", &BACKEND_UNBUILT),
        Error::NotCompiledWithMetalSupport => ("NotCompiledWithMetalSupport", &BACKEND_UNBUILT),
        _ => ("unnamed", &UNTOLD),
    }
}

/// The variant of `error`, as a refusal names it.
pub fn variant(error: &Error) -> &'static str {
    classify(error).0
}

/// The refusal of `step` (`computing the forward pass of pair 3`), which
/// Candle failed with `error`: the declared refusal Ster's own tensor code
/// raised inside it ([`super::refuse_candle`]), or the cause its variant
/// declares.
pub fn computing(step: &str, error: Error) -> anyhow::Error {
    let error = match candle::lift(error) {
        Ok(declared) => return declared,
        Err(error) => error,
    };
    let (variant, cause) = classify(&error);
    let shown = error.to_string();
    refuse_from(
        cause,
        &[
            ("step", step),
            ("variant", variant),
            ("error", &shown),
            ("within", "Candle raised it at this step itself"),
        ],
        error.into(),
    )
}

/// `error` as a command ends with it: [`within`] the command `command`
/// (`ster tune dpo`, `ster request tune/dpo`). Every `?` on a Candle result
/// that no site declared ends here.
pub fn settle(error: anyhow::Error, command: &str) -> anyhow::Error {
    within(error, &format!("running `{command}`"))
}

/// `error`, raised while Ster was at `step` (`scoring the examples`): itself
/// when a cause declares it, with `step` added when it holds no Candle error
/// either, else the cause the Candle error's variant declares, showing the
/// context lines the chain carries above that error as where it was raised.
pub fn within(error: anyhow::Error, step: &str) -> anyhow::Error {
    if declared(&error).is_some() {
        return error;
    }
    let mut above: Vec<String> = Vec::new();
    let mut found = None;
    for link in error.chain() {
        match link.downcast_ref::<Error>() {
            Some(candle_error) => {
                found = Some(candle_error);
                break;
            }
            None => above.push(link.to_string()),
        }
    }
    let Some(candle_error) = found else {
        return super::amend(error, format!("it was {step}"));
    };
    // A refusal Ster's own tensor code raised travels inside Candle's error
    // (`refuse_candle`), where the chain does not reach it.
    let inner = match candle::root(candle_error) {
        Error::WrappedContext { wrapped, .. } => wrapped.downcast_ref::<Declared>(),
        _ => None,
    };
    if let Some(inner) = inner {
        return anyhow::Error::new(Declared {
            cause: inner.cause,
            sentence: format!("{}; it was {step}", inner.sentence),
            fix: inner.fix.clone(),
            underneath: None,
        });
    }
    let within = if above.is_empty() {
        "no context above it says where in the step it was raised".to_owned()
    } else {
        format!("raised within: {}", above.join(": "))
    };
    let (variant, cause) = classify(candle_error);
    let shown = candle_error.to_string();
    refuse_from(
        cause,
        &[
            ("step", step),
            ("variant", variant),
            ("error", &shown),
            ("within", &within),
        ],
        error,
    )
}
