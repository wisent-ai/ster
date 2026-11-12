//! A tensor a checkpoint stores in another shape than its config gives it,
//! raised the way Candle raises its own (`Error::UnexpectedShape`), so the
//! load refuses it as `ster.model.weights.weights-unfit`, naming the tensor
//! and both shapes, whichever family's reading found it.

use candle_core::Shape;

/// The refusal of `name`, stored as `got` where the config sizes it
/// `expected`.
pub(crate) fn misshapen(name: &str, expected: impl Into<Shape>, got: &Shape) -> candle_core::Error {
    candle_core::Error::UnexpectedShape {
        msg: format!("shape mismatch for {name}"),
        expected: expected.into(),
        got: got.clone(),
    }
    .bt()
}
