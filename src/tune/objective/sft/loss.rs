//! The loss of one supervised example: next-token cross-entropy over the
//! completion only.

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::loss;

use crate::runtime::Runtime;

/// `logits` is `[1, n, vocab]`, and the distribution that predicts token `t`
/// sits at the logit position of the token before it. Scoring the completion
/// therefore means the window that starts at the last prompt token and is as
/// long as the completion, against the targets `ids[boundary..]`. A
/// completion needs a prompt token before it, and a token to predict: the
/// trainer refuses an example that encodes without either before its first
/// step, so both hold here.
pub(super) fn completion_loss(
    logits: &Tensor,
    ids: &[u32],
    boundary: usize,
    runtime: &Runtime,
) -> Result<Tensor> {
    assert!(
        boundary > 0 && boundary < ids.len(),
        "an example reaches the loss with a prompt token and a completion token"
    );
    let start = boundary - 1;
    let completion = &ids[boundary..];
    let (_, positions, vocab) = logits.dims3()?;
    assert_eq!(
        positions,
        ids.len(),
        "the forward pass answers one position per token"
    );
    // Scored in F32 whatever the base weights are. `cross_entropy` takes the
    // log-softmax of this window, and a half-precision log-softmax over a
    // vocabulary is the one place in this loss where the summation range
    // genuinely does not fit. `to_dtype` is differentiable and returns the
    // gradient in the argument's dtype, so this costs a clone at F32 and
    // nothing else.
    let window = logits
        .i((.., start..start + completion.len(), ..))?
        .reshape((completion.len(), vocab))?
        .to_dtype(DType::F32)?;
    let targets = Tensor::new(completion, runtime.device())?;
    Ok(loss::cross_entropy(&window, &targets)?)
}
