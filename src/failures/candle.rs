//! Declared refusals raised where Candle's `Result` is the type: a decoder
//! built from a checkpoint's config and weights. Candle has no variant for
//! another crate's error but `Error::WrappedContext`, so the refusal travels
//! boxed there, and [`lift`] takes it back out where the build's error
//! becomes Ster's (`failures::model::weights_unfit`).

use wisent_errors::Cause;

use super::Declared;

/// The refusal `cause` declares, its `{name}` placeholders filled from
/// `values`, as a Candle error.
pub fn refuse_candle(cause: &'static Cause, values: &[(&str, &str)]) -> candle_core::Error {
    let declared = Declared {
        cause,
        sentence: cause.sentence(values),
        fix: cause.fix(values),
        underneath: None,
    };
    candle_core::Error::WrappedContext {
        wrapped: Box::new(declared),
        context: cause.id.to_owned(),
    }
}

/// The declared refusal [`refuse_candle`] raised inside `error`, under any
/// backtrace, context or path Candle added on its way out; `Err(error)`
/// unchanged when it carries none.
pub fn lift(error: candle_core::Error) -> Result<anyhow::Error, candle_core::Error> {
    use candle_core::Error;
    match error {
        Error::WrappedContext { wrapped, context } => match wrapped.downcast::<Declared>() {
            Ok(declared) => Ok(anyhow::Error::new(*declared)),
            Err(wrapped) => Err(Error::WrappedContext { wrapped, context }),
        },
        Error::WithBacktrace { inner, backtrace } => lift(*inner).map_err(|inner| {
            Error::WithBacktrace {
                inner: Box::new(inner),
                backtrace,
            }
        }),
        Error::Context { inner, context } => lift(*inner).map_err(|inner| Error::Context {
            inner: Box::new(inner),
            context,
        }),
        Error::WithPath { inner, path } => lift(*inner).map_err(|inner| Error::WithPath {
            inner: Box::new(inner),
            path,
        }),
        other => Err(other),
    }
}

/// `error` as Ster's error: the declared refusal [`refuse_candle`] raised
/// inside it, or Candle's own error where it carries none.
pub fn lifted(error: candle_core::Error) -> anyhow::Error {
    match lift(error) {
        Ok(declared) => declared,
        Err(error) => anyhow::Error::new(error),
    }
}

/// The error Candle raised, under any backtrace, context or path it added on
/// its way out: what a refusal tells its reasons apart by.
pub fn root(error: &candle_core::Error) -> &candle_core::Error {
    use candle_core::Error;
    match error {
        Error::WithBacktrace { inner, .. }
        | Error::Context { inner, .. }
        | Error::WithPath { inner, .. } => root(inner),
        other => other,
    }
}

/// Whether `error` says two tensors' shapes do not fit each other.
pub fn shapes_differ(error: &candle_core::Error) -> bool {
    use candle_core::Error;
    matches!(
        root(error),
        Error::ShapeMismatchBinaryOp { .. }
            | Error::ShapeMismatch { .. }
            | Error::UnexpectedShape { .. }
            | Error::ShapeMismatchCat { .. }
            | Error::BroadcastIncompatibleShapes { .. }
            | Error::UnexpectedNumberOfDims { .. }
    )
}
