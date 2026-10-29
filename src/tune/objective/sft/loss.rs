//! The loss of one supervised example: next-token cross-entropy over the
//! completion only.

use anyhow::{Result, bail};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::loss;

use crate::runtime::Runtime;

/// `logits` is `[1, n, vocab]`, and the distribution that predicts token `t`
/// sits at the logit position of the token before it. Scoring the completion
/// therefore means the window that starts at the last prompt token and is as
/// long as the completion, against the targets `ids[boundary..]`. A
/// completion needs a prompt token before it, and `encode_example` always
/// prepends the prompt's special tokens.
pub(super) fn completion_loss(
    logits: &Tensor,
    ids: &[u32],
    boundary: usize,
    runtime: &Runtime,
) -> Result<Tensor> {
    let (Some(prompt), Some(completion)) = (ids.get(..boundary), ids.get(boundary..)) else {
        bail!(
            "a training example's completion starts at token {boundary} of {}",
            ids.len()
        );
    };
    let Some((start, _)) = prompt.iter().enumerate().last() else {
        bail!("a training example must begin with at least one prompt token");
    };
    if completion.is_empty() {
        bail!("a training example must end with at least one completion token");
    }
    let (_, positions, vocab) = logits.dims3()?;
    if positions != ids.len() {
        bail!(
            "the forward pass returned {positions} positions for {} tokens",
            ids.len()
        );
    }
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
