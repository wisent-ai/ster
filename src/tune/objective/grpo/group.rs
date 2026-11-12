//! One prompt's group: the conversations drawn for it, the advantage each one
//! carries against the group's own mean, and the loss the group produces.

use anyhow::Result;
use candle_core::Tensor;

use crate::{
    failures::{
        amend, refuse,
        tune::{
            objectives::{GROUP_TOO_SMALL, TURNS_NONE},
            policy::{GROUP_SPREAD_NOT_FINITE, REWARD_NOT_FINITE},
            run::nothing_fits,
        },
    },
    model::Route,
    runtime::{Completion, Runtime},
    workflow::{self, PromptSet},
};

use super::super::super::preflight::token_logprobs;
use super::rollout::{UserSimulator, rollout};
use super::{GrpoOptions, Reward};

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
    for conversation_index in 0..options.group {
        let conversation = rollout(
            runtime,
            user,
            prompt,
            options.turns,
            options.generation,
            limit,
            draw,
        )?;
        let mut total = 0f64;
        for (turn, (completion, &before)) in conversation
            .turns
            .iter()
            .zip(&conversation.before)
            .enumerate()
        {
            let score = if user.is_some() && matches!(reward, Reward::Scorer(_)) {
                let context = crate::json::compact(&conversation.transcript[..before]);
                reward.score(&context, completion)?
            } else {
                reward.score(prompt, completion)?
            };
            total += score;
            if !total.is_finite() {
                return Err(refuse(
                    &REWARD_NOT_FINITE,
                    &[
                        ("conversation", &(conversation_index + 1).to_string()),
                        ("group", &options.group.to_string()),
                        ("turn", &(turn + 1).to_string()),
                        ("score", &score.to_string()),
                        ("total", &total.to_string()),
                        ("source", source(reward)),
                    ],
                ));
            }
        }
        rewards.push(total);
        conversations.push(conversation.turns);
    }

    let count = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / count;
    let variance = rewards
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count;
    if !mean.is_finite() || !variance.is_finite() {
        let listed: Vec<String> = rewards.iter().map(f64::to_string).collect();
        return Err(refuse(
            &GROUP_SPREAD_NOT_FINITE,
            &[
                ("count", &rewards.len().to_string()),
                ("rewards", &listed.join(", ")),
                ("mean", &mean.to_string()),
                ("variance", &variance.to_string()),
                ("source", source(reward)),
            ],
        ));
    }
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

/// What `--reward` chose, as a refusal of its rewards names it.
fn source(reward: &Reward) -> &'static str {
    match reward {
        Reward::Length => "the length reward",
        Reward::Model(_) => "a reward artifact's head",
        Reward::Scorer(_) => "an outside scorer",
    }
}

/// The loss for one group, plus the scalars the report is built from.
pub(super) struct Loss {
    pub(super) tensor: Tensor,
    pub(super) value: f64,
    pub(super) kl: f64,
}

pub(super) fn group_loss(runtime: &Runtime, group: &Group, options: &GrpoOptions) -> Result<Loss> {
    // The group is the baseline: fewer than two conversations compare
    // nothing, so the check `ster tune grpo` makes of --group is made again
    // of the group actually drawn.
    if group.draws.len() < 2 {
        return Err(refuse(
            &GROUP_TOO_SMALL,
            &[("group", &group.draws.len().to_string())],
        ));
    }
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
        // `rollout` refuses a draw whose first assistant turn is empty
        // (`turn-empty`) and `--turns` is at least one (`turns-none`), so
        // every draw holds a turn and the loop above summed it.
        let conversation = conversation
            .expect("every draw holds a turn: rollout refuses one without, --turns is at least 1");
        kl_total += conversation_kl / draw.turns.len() as f64;
        let scaled = (conversation.neg()? / group.draws.len() as f64)?;
        summed = Some(match summed {
            Some(total) => (total + scaled)?,
            None => scaled,
        });
    }
    let Some(tensor) = summed else {
        return Err(refuse(&GROUP_TOO_SMALL, &[("group", "0")]));
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

/// The indices of `prompts` whose prompt plus its longest possible completion
/// fits in `limit` tokens. Each that does not is skipped with a progress line,
/// and a set none of whose prompts fits is refused. The prompt is tokenized
/// once here only to decide that; the sampler tokenizes it again per draw and
/// that encode is the one the scored sequence comes from.
pub(super) fn fitting_prompts(
    runtime: &Runtime,
    prompts: &PromptSet,
    max_new_tokens: usize,
    limit: usize,
) -> Result<Vec<usize>> {
    let mut usable: Vec<usize> = Vec::with_capacity(prompts.prompts.len());
    let mut lengths = Vec::with_capacity(prompts.prompts.len());
    for (index, prompt) in prompts.prompts.iter().enumerate() {
        let ids = runtime.encode(prompt).map_err(|error| {
            amend(
                error,
                format!("it was prompt {index} of {}", prompts.prompts.len()),
            )
        })?;
        lengths.push(ids.len() + max_new_tokens);
        if ids.len() + max_new_tokens > limit {
            workflow::progress(format!(
                "skipping prompt {index}: {} prompt tokens plus {max_new_tokens} sampled tokens exceed the {limit} token limit",
                ids.len()
            ));
            continue;
        }
        usable.push(index);
    }
    if usable.is_empty() {
        return Err(nothing_fits("prompt", "train on", lengths, limit));
    }
    Ok(usable)
}
