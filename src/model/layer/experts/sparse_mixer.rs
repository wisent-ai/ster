//! PhiMoE's SparseMixer routing.

use candle_core::{D, Tensor};

/// PhiMoE's SparseMixer as Transformers' `sparsemixer` runs it outside
/// training. Each of `rounds` rounds takes the highest logit not yet chosen
/// (the first on a tie), drops the chosen experts and every expert whose
/// distance below it, divided by the larger of its own magnitude and the
/// best logit, exceeds `2 · jitter`, and weighs the chosen expert by its
/// softmax over the rest. The choice is read on the host; each weight is a
/// softmax of the logits through masks, so the router keeps its gradient.
pub(super) fn sparse_mixer(
    logits: &Tensor,
    rounds: usize,
    jitter: f32,
    routed: &mut [Vec<u32>],
) -> candle_core::Result<Tensor> {
    let (tokens, count) = logits.dims2()?;
    let host = logits.to_vec2::<f32>()?;
    // Per round: an additive mask, zero for an expert kept and minus
    // infinity for one dropped, and the chosen expert marked with a one.
    let mut dropped = vec![vec![0f32; tokens * count]; rounds];
    let mut picked = vec![vec![0f32; tokens * count]; rounds];
    for (token, row) in host.iter().enumerate() {
        let mut taken = vec![false; count];
        for (drop, pick) in dropped.iter_mut().zip(picked.iter_mut()) {
            let Some(best) = (0..count)
                .filter(|expert| !taken[*expert])
                .reduce(|kept, next| if row[next] > row[kept] { next } else { kept })
            else {
                break;
            };
            let top = row[best];
            for (expert, logit) in row.iter().enumerate() {
                if taken[expert] || (top - logit) / logit.abs().max(top) > 2.0 * jitter {
                    drop[token * count + expert] = f32::NEG_INFINITY;
                }
            }
            pick[token * count + best] = 1.0;
            routed[best].push(token as u32);
            taken[best] = true;
        }
    }
    let device = logits.device();
    let mut weights = logits.zeros_like()?;
    for (drop, pick) in dropped.into_iter().zip(picked) {
        let drop = Tensor::from_vec(drop, (tokens, count), device)?;
        let pick = Tensor::from_vec(pick, (tokens, count), device)?;
        let gates = candle_nn::ops::softmax(&(logits + drop)?, D::Minus1)?;
        weights = (weights + (gates * pick)?)?;
    }
    Ok(weights)
}
