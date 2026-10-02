//! What one preference step costs: the frozen reference's opinion of both
//! sides, the policy's log-ratio against it, and the loss that falls out.

use anyhow::Result;
use candle_core::Tensor;

use crate::{model::Route, runtime::Runtime, workflow};

use super::super::super::{
    batch,
    preflight::{EncodedPair, sequence_logprob, softplus},
};
use super::{DpoLoss, DpoOptions};

/// One tokenized pair with the frozen reference's opinion of both sides.
pub(super) struct Scored {
    pub(super) pair: EncodedPair,
    pub(super) chosen_reference: f64,
    pub(super) rejected_reference: f64,
}

impl Scored {
    /// The reference values are filled in by [`reference_scores`] before the
    /// optimizer exists; zero is not a plausible log-probability and would
    /// surface immediately as a first loss that is not `ln 2`.
    pub(super) fn new(pair: EncodedPair) -> Self {
        Self {
            pair,
            chosen_reference: 0.0,
            rejected_reference: 0.0,
        }
    }
}

/// Fills in each pair's frozen-reference log-probabilities.
///
/// Run once, before the optimizer exists. The reference never changes, so
/// re-deriving these every epoch would be `2 * pairs * (epochs - 1)` forward
/// passes spent recomputing constants.
pub(super) fn reference_scores(
    runtime: &Runtime,
    encoded: &mut [Scored],
    pairs: usize,
) -> Result<()> {
    let total = encoded.len();
    workflow::progress(format!(
        "scoring {total} pairs under the frozen reference model"
    ));
    let device = runtime.device();
    let mut scored = 0usize;
    // Batched on the same knob the training loop uses, in input order: the
    // reference is a constant and nothing here is shuffled, so the only thing
    // a group decides is how many rows share a kernel launch.
    for group in encoded.chunks_mut(pairs.max(1)) {
        let mut rows: Vec<&[u32]> = Vec::with_capacity(group.len() * 2);
        for pair in group.iter() {
            rows.push(&pair.pair.chosen);
            rows.push(&pair.pair.rejected);
        }
        let read = batch::read_rows(&rows, pairs, 2, |pass| {
            runtime.forward_scored_rows(pass, Route::Base)
        })?;
        for (position, pair) in group.iter_mut().enumerate() {
            pair.chosen_reference =
                sequence_logprob(&read[position * 2], &pair.pair.chosen, 1, device)?
                    .to_scalar::<f32>()? as f64;
            pair.rejected_reference =
                sequence_logprob(&read[position * 2 + 1], &pair.pair.rejected, 1, device)?
                    .to_scalar::<f32>()? as f64;
            scored += 1;
            workflow::progress(format!("reference pair {scored}/{total}"));
        }
    }
    Ok(())
}

/// The loss for one pair, plus the scalars the report is built from.
pub(super) struct Step {
    pub(super) tensor: Tensor,
    pub(super) loss: f64,
    pub(super) chosen_reward: f64,
    pub(super) rejected_reward: f64,
}

/// One pair's contribution: two scored rows, one margin, one loss.
///
/// The logits arrive already read out of whatever forward produced them, so
/// this function is identical whether one pair went through the model or
/// eight did.
pub(super) fn step_loss(
    runtime: &Runtime,
    scored: &Scored,
    chosen_logits: &Tensor,
    rejected_logits: &Tensor,
    options: &DpoOptions,
) -> Result<Step> {
    let chosen = policy_log_ratio(
        runtime,
        chosen_logits,
        &scored.pair.chosen,
        scored.chosen_reference,
        options,
    )?;
    let rejected = policy_log_ratio(
        runtime,
        rejected_logits,
        &scored.pair.rejected,
        scored.rejected_reference,
        options,
    )?;
    let margin = (&chosen - &rejected)?;
    let tensor = match options.loss {
        // -log sigmoid(beta * margin), written as the softplus that does not
        // overflow when the policy is already confident.
        DpoLoss::Dpo => softplus(&(&margin * -options.beta)?)?,
        // Equation 17 of the IPO paper: a squared error against the fixed
        // target 1 / (2 * beta) rather than a sigmoid, which is what stops the
        // objective from being satisfied by driving the margin to infinity.
        DpoLoss::Ipo => (&margin - 1.0 / (2.0 * options.beta))?.sqr()?,
    };
    // The implicit reward DPO derives is beta times the log-ratio. Reading it
    // off the same two tensors the loss was built from — rather than
    // recomputing it — is what makes the reported accuracy the accuracy of the
    // step that was actually taken.
    Ok(Step {
        loss: tensor.to_scalar::<f32>()? as f64,
        chosen_reward: options.beta * chosen.to_scalar::<f32>()? as f64,
        rejected_reward: options.beta * rejected.to_scalar::<f32>()? as f64,
        tensor,
    })
}

/// The policy's log-probability of `ids` minus the reference's, normalized for
/// the objective that will consume it.
fn policy_log_ratio(
    runtime: &Runtime,
    logits: &Tensor,
    ids: &[u32],
    reference: f64,
    options: &DpoOptions,
) -> Result<Tensor> {
    let policy = sequence_logprob(logits, ids, 1, runtime.device())?;
    let ratio = (policy - reference)?;
    if options.loss.length_normalized() {
        // `sequence_logprob` scores every token after the begin-of-sequence
        // marker, so that is the count the mean divides by.
        return Ok((ratio / (ids.len() - 1) as f64)?);
    }
    Ok(ratio)
}

/// Running totals over one epoch.
#[derive(Debug, Default)]
pub(super) struct Summary {
    pub(super) pairs: usize,
    pub(super) loss: f64,
    pub(super) correct: usize,
    pub(super) chosen_reward: f64,
    pub(super) rejected_reward: f64,
}

impl Summary {
    pub(super) fn record(&mut self, step: &Step) {
        self.pairs += 1;
        self.loss += step.loss;
        self.chosen_reward += step.chosen_reward;
        self.rejected_reward += step.rejected_reward;
        if step.chosen_reward > step.rejected_reward {
            self.correct += 1;
        }
    }

    /// Every mean below divides by the pair count, guarded at one so an epoch
    /// that recorded nothing reports zero rather than a JSON `NaN` no client
    /// can parse.
    fn mean(&self, total: f64) -> f32 {
        (total / self.pairs.max(1) as f64) as f32
    }

    pub(super) fn mean_loss(&self) -> f32 {
        self.mean(self.loss)
    }

    pub(super) fn mean_chosen(&self) -> f32 {
        self.mean(self.chosen_reward)
    }

    pub(super) fn mean_rejected(&self) -> f32 {
        self.mean(self.rejected_reward)
    }

    pub(super) fn mean_margin(&self) -> f32 {
        self.mean(self.chosen_reward - self.rejected_reward)
    }

    pub(super) fn accuracy(&self) -> f32 {
        self.correct as f32 / self.pairs.max(1) as f32
    }
}
