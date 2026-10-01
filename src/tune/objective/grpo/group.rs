//! One prompt's group: the conversations drawn for it, the advantage each one
//! carries against the group's own mean, and the loss the group produces.

use anyhow::{Result, bail};
use candle_core::Tensor;

use crate::{
    model::Route,
    runtime::{Completion, Runtime},
};

use super::super::super::preflight::token_logprobs;
use super::rollout::{UserSimulator, rollout};
use super::{GrpoIteration, GrpoOptions, Reward};

/// One assistant turn with the frozen reference's view of it.
pub(super) struct Turn {
    pub(super) completion: Completion,
    /// The frozen reference's per-token log-probabilities of this turn.
    /// Constant: the reference cannot move, and it is scored here rather than
    /// inside the loss so the tensor carries no autograd tape.
    pub(super) reference: Tensor,
}

/// One sampled conversation with everything the step needs about it.
pub(super) struct Draw {
    pub(super) turns: Vec<Turn>,
    pub(super) advantage: f64,
}

/// A whole group for one prompt.
pub(super) struct Group {
    pub(super) draws: Vec<Draw>,
    pub(super) mean_reward: f64,
    pub(super) spread: f64,
}

/// Draws `--group` conversations, scores them, and normalizes within the
/// group.
///
/// A conversation's reward is the sum of its assistant turns' rewards, each
/// turn scored with the conversation before it in front. A conversation the
/// simulated user ends early has fewer turns to earn from, which is how the
/// length of the conversation in turns enters the reward without a separate
/// term anyone has to weigh against the reward head.
pub(super) fn sample_group(
    runtime: &Runtime,
    user: Option<&UserSimulator>,
    prompt: &str,
    reward: &Reward,
    options: &GrpoOptions,
    limit: usize,
    draw: &mut u64,
) -> Result<Group> {
    let mut conversations = Vec::with_capacity(options.group);
    let mut rewards = Vec::with_capacity(options.group);
    for _ in 0..options.group {
        let turns = rollout(runtime, user, prompt, options, limit, draw)?;
        let mut total = 0f64;
        for turn in &turns {
            total += reward.score(turn)?;
        }
        rewards.push(total);
        conversations.push(turns);
    }

    let count = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / count;
    let variance = rewards
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count;
    let spread = variance.sqrt();
    // No epsilon. A group whose conversations all scored the same has no
    // preference to express, and its advantages are exactly zero; adding a
    // floor to the denominator would turn that silence into amplified rounding.
    let normalize = |value: f64| {
        if spread > 0.0 {
            (value - mean) / spread
        } else {
            0.0
        }
    };

    let mut draws = Vec::with_capacity(conversations.len());
    for (completions, value) in conversations.into_iter().zip(&rewards) {
        let mut turns = Vec::with_capacity(completions.len());
        for completion in completions {
            let ids = sequence(&completion);
            let logits = runtime.forward_scored(&ids, Route::Base)?;
            let reference =
                token_logprobs(&logits, &ids, completion.prompt.len(), runtime.device())?;
            turns.push(Turn {
                completion,
                reference,
            });
        }
        draws.push(Draw {
            turns,
            advantage: normalize(*value),
        });
    }
    Ok(Group {
        draws,
        mean_reward: mean,
        spread,
    })
}

/// The loss for one group, plus the scalars the report is built from.
pub(super) struct Loss {
    pub(super) tensor: Tensor,
    pub(super) value: f64,
    pub(super) kl: f64,
}

pub(super) fn group_loss(runtime: &Runtime, group: &Group, options: &GrpoOptions) -> Result<Loss> {
    let mut summed: Option<Tensor> = None;
    let mut kl_total = 0f64;
    for draw in &group.draws {
        let mut conversation: Option<Tensor> = None;
        let mut conversation_kl = 0f64;
        for turn in &draw.turns {
            let ids = sequence(&turn.completion);
            let logits = runtime.forward_train(&ids)?;
            let policy = token_logprobs(
                &logits,
                &ids,
                turn.completion.prompt.len(),
                runtime.device(),
            )?;

            // pi_old is pi_theta at this exact step, so the ratio is one in
            // value and its gradient is the policy gradient. Detaching is what
            // states that, and it costs no second forward pass.
            let ratio = (&policy - policy.detach())?.exp()?;
            let advantage = (ratio * draw.advantage)?;

            // k3: exp(d) - d - 1 with d = log pi_ref - log pi_theta.
            // Non-negative for every sample and unbiased for the divergence,
            // where the naive -d is neither.
            let divergence = (&turn.reference - &policy)?;
            let penalty = ((divergence.exp()? - &divergence)? - 1.0)?;
            conversation_kl += penalty.mean_all()?.to_scalar::<f32>()? as f64;

            // Averaged over the turn's own tokens, then over the turns, so a
            // long turn does not outvote a short one on length alone.
            let objective = (advantage - (penalty * options.beta)?)?.mean_all()?;
            let scaled = (objective / draw.turns.len() as f64)?;
            conversation = Some(match conversation {
                Some(total) => (total + scaled)?,
                None => scaled,
            });
        }
        let Some(conversation) = conversation else {
            bail!("a sampled conversation contained no assistant turns");
        };
        kl_total += conversation_kl / draw.turns.len() as f64;
        let scaled = (conversation.neg()? / group.draws.len() as f64)?;
        summed = Some(match summed {
            Some(total) => (total + scaled)?,
            None => scaled,
        });
    }
    let Some(tensor) = summed else {
        bail!("a sampled group contained no conversations");
    };
    Ok(Loss {
        value: tensor.to_scalar::<f32>()? as f64,
        kl: kl_total / group.draws.len() as f64,
        tensor,
    })
}

/// Prompt then completion, the exact sequence the sampler produced.
fn sequence(completion: &Completion) -> Vec<u32> {
    let mut ids = Vec::with_capacity(completion.prompt.len() + completion.tokens.len());
    ids.extend_from_slice(&completion.prompt);
    ids.extend_from_slice(&completion.tokens);
    ids
}

/// Running totals over one iteration.
#[derive(Debug, Default)]
pub(super) struct Totals {
    pub(super) groups: usize,
    pub(super) completions: usize,
    pub(super) turns: usize,
    pub(super) reward: f64,
    pub(super) spread: f64,
    pub(super) kl: f64,
    pub(super) loss: f64,
    pub(super) tokens: usize,
}

impl Totals {
    pub(super) fn record(&mut self, group: &Group, loss: &Loss) {
        self.groups += 1;
        self.completions += group.draws.len();
        self.turns += group
            .draws
            .iter()
            .map(|draw| draw.turns.len())
            .sum::<usize>();
        self.reward += group.mean_reward;
        self.spread += group.spread;
        self.kl += loss.kl;
        self.loss += loss.value;
        self.tokens += group
            .draws
            .iter()
            .flat_map(|draw| &draw.turns)
            .map(|turn| turn.completion.tokens.len())
            .sum::<usize>();
    }

    /// Every mean divides by the group count, guarded at one so an iteration
    /// that recorded nothing reports zero rather than a JSON `NaN` no client
    /// can parse.
    pub(super) fn mean(&self, total: f64) -> f32 {
        (total / self.groups.max(1) as f64) as f32
    }

    pub(super) fn mean_reward(&self) -> f32 {
        self.mean(self.reward)
    }

    pub(super) fn mean_kl(&self) -> f32 {
        self.mean(self.kl)
    }

    pub(super) fn finish(&self, iteration: usize) -> GrpoIteration {
        GrpoIteration {
            iteration,
            groups: self.groups,
            completions: self.completions,
            mean_reward: self.mean_reward(),
            reward_spread: self.mean(self.spread),
            mean_kl: self.mean_kl(),
            policy_loss: self.mean(self.loss),
            mean_turns: self.turns as f32 / self.completions.max(1) as f32,
            mean_completion_tokens: self.tokens as f32 / self.turns.max(1) as f32,
        }
    }
}
