//! Group-relative policy optimization.
//!
//! The other trainers here learn from text someone already wrote. This one
//! learns from text the policy writes: for each prompt it draws a group of
//! completions, scores them, and pushes the policy toward the ones that scored
//! above their own group's average. There is no target sequence anywhere in it.
//!
//! The pieces that are decisions rather than details:
//!
//! * **The baseline is the group.** Classic policy gradient needs a value model
//!   to tell it whether a reward was good. GRPO does not: it samples several
//!   completions for the same prompt and uses their mean as the baseline, which
//!   removes an entire second network and makes the advantage scale-free. What
//!   it costs is `--group` generations per prompt, and that is the dominant
//!   cost of the whole loop.
//!
//! * **A degenerate group teaches nothing, and says so quietly.** If every
//!   completion in a group scores the same, the deviations are all zero and so
//!   are the advantages, and the group contributes only its KL term. That falls
//!   out of the arithmetic rather than being special-cased, and it is why an
//!   epsilon in the denominator would be wrong: it would turn "no signal" into
//!   "amplify the rounding".
//!
//! * **The ratio is exactly one, and is written anyway.** GRPO's objective
//!   carries the importance ratio `pi_theta / pi_old`. With one gradient step
//!   per sampling round — which is what this implements — `pi_old` *is*
//!   `pi_theta` at the moment of the step, so the ratio is exactly one in value
//!   and its gradient is exactly the policy gradient. Writing it as
//!   `exp(logp - logp.detach())` rather than collapsing it to `logp * A` costs
//!   one exponential, needs no second forward pass to recover `logp_old`, and
//!   keeps the reported loss on the same scale as the published objective:
//!   `-A + beta * KL` rather than an unbounded log-probability.
//!
//! * **The KL is the k3 estimator.** `exp(d) - d - 1` for `d = logp_ref -
//!   logp_theta` is non-negative for every sample and unbiased for the
//!   divergence, where the naive `-d` is neither. The reference is the frozen
//!   base — the same weights with the adapters skipped — for the same reason
//!   preference optimization uses it: it is already mapped.
//!
//! A run given a checkpoint directory writes its state there every
//! `--checkpoint-every` steps, and a run given `--resume` continues from the
//! newest one exactly where it stopped: the draw counter that seeds every
//! sample, the optimizer's moments and steps, and the totals of the iteration
//! in progress are the ones the stopped run had.

use std::path::Path;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{ParamsAdamW, VarMap};

use super::super::{
    checkpoint::{Position, optimizer::AdamW},
    preflight::Trainable,
    schedule,
};
use crate::{
    failures::amend,
    runtime::Runtime,
    workflow::{self, PromptSet},
};

mod group;
mod reward;
mod rollout;
mod running;
mod totals;

pub use reward::{GrpoIteration, GrpoOptions, GrpoReport, Reward};
pub use rollout::{Rollout, UserSimulator, rollout};

use group::{fitting_prompts, group_loss, sample_group};
use running::{Running, SUBJECT};
use totals::Totals;

/// Runs the loop against `prompts`, scoring with `reward`.
///
/// `runtime` must be the one `Runtime::load_trainable` returned alongside
/// `varmap`. It plays three roles at once and can, because they differ only in
/// which forward mode they ask for: it is the policy being sampled from, the
/// policy being scored with a gradient, and — with the adapters skipped — the
/// frozen reference the KL is measured against. `user` writes the user's
/// turns when `options.turns` is above one, and must be absent otherwise.
pub fn grpo(
    runtime: &Runtime,
    varmap: &VarMap,
    prompts: &PromptSet,
    reward: &Reward,
    requested_reward: &str,
    user: Option<(&UserSimulator, &str)>,
    options: &GrpoOptions,
) -> Result<GrpoReport> {
    options.check(user.map(|(_, model)| model))?;
    let max_new_tokens = options.generation.max_new_tokens;
    let simulator = user.map(|(simulator, _)| simulator);
    // The judge is a different model from the policy, but it reads the same
    // kind of residual stream, and it was fitted in whatever encoding and
    // precision trained it. A reward artifact from an f16 run scoring an f32
    // policy is reading a space it was not fitted in, and the reward it
    // returns looks perfectly ordinary, so the run says so.
    if let Some(path) = matches!(reward, Reward::Model(_)).then_some(requested_reward) {
        crate::tune::warn_on_provenance(Path::new(path), "reward model", runtime);
    }
    let Trainable {
        spec,
        vars: _,
        tensors,
        parameters,
        limit,
    } = options.preflight().open(runtime, varmap, &options.spec)?;

    let usable = fitting_prompts(runtime, prompts, max_new_tokens, limit)?;

    let fingerprint = options.fingerprint(requested_reward, user.map(|(_, model)| model));
    let mut optimizer = AdamW::over(
        varmap,
        ParamsAdamW {
            lr: options.learning_rate,
            ..Default::default()
        },
    )?;
    let resumed = options.durable.load(&mut optimizer, &fingerprint)?;
    // Every draw gets its own seed, advanced from the operator's and counted in
    // `running.draw`: a fixed seed would draw one completion `--group` times
    // over, and a group with no variety has no baseline.
    let mut running: Running = match &resumed {
        Some(position) => position.running(SUBJECT)?,
        None => Running::default(),
    };
    let mut checkpoints = Vec::new();
    let steps: Vec<&[usize]> = usable.chunks(options.accumulation).collect();
    let total_steps = steps.len() * options.iterations;

    for (iteration, _) in std::iter::repeat_n((), options.iterations).enumerate() {
        let (mut rest, mut totals) = match &resumed {
            Some(position) if position.pass > iteration => continue,
            Some(position) if position.pass == iteration => (
                position.remaining(&steps, SUBJECT)?,
                Totals::from(&running.totals),
            ),
            _ => (&steps[..], Totals::default()),
        };

        while let Some((group_slots, after)) = rest.split_first() {
            rest = after;
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                optimizer.rates.len(),
                total_steps,
                options.warmup_steps,
            ));
            let mut summed: Option<Tensor> = None;
            let mut losses = Vec::with_capacity(group_slots.len());
            for &index in group_slots.iter() {
                workflow::progress(format!(
                    "iterations done {}/{} prompt {index} sampling {} completions",
                    running.history.len(),
                    options.iterations,
                    options.group
                ));
                let prompt = &prompts.prompts[index];
                let group = sample_group(
                    runtime,
                    simulator,
                    prompt,
                    reward,
                    options,
                    limit,
                    &mut running.draw,
                )
                .map_err(|error| {
                    optimizer.failed(amend(
                        error,
                        format!("it was sampling the group of prompt {index}"),
                    ))
                })?;
                let loss = group_loss(runtime, &group, options).map_err(|error| {
                    optimizer.failed(amend(error, format!("it was the loss of prompt {index}")))
                })?;
                totals.record(&group, &loss);
                losses.push(loss.value);
                let scaled = (loss.tensor / options.accumulation as f64)
                    .and_then(|scaled| match summed.take() {
                        Some(total) => total + scaled,
                        None => Ok(scaled),
                    })
                    .map_err(|error| optimizer.failed(error.into()))?;
                summed = Some(scaled);
            }
            optimizer.backward_summed(summed, group_slots.len(), group_slots.len(), "prompt")?;

            let group_mean = (losses.iter().sum::<f64>() / losses.len() as f64) as f32;
            running.first_loss.get_or_insert(group_mean);
            running.final_loss = Some(group_mean);
            let step = optimizer.rates.len();
            workflow::progress(format!(
                "iterations done {}/{} step {step}/{total_steps} loss {group_mean:.4} reward {:.4} kl {:.5}",
                running.history.len(),
                options.iterations,
                totals.mean_reward(),
                totals.mean_kl()
            ));
            if options.durable.due(step) {
                running.totals = (&totals).into();
                let within = steps.len() - rest.len();
                let position = Position {
                    pass: iteration,
                    step,
                    within,
                    running: crate::json::value(&running),
                };
                options.durable.save(&optimizer, &fingerprint, &position)?;
                checkpoints.push(step);
            }
        }

        let done = running.history.len() + 1;
        let mut summary = totals.finish(iteration);
        // The report counts iterations from one, as an operator does.
        summary.iteration = done;
        workflow::progress(format!(
            "iterations done {done}/{} mean reward {:.4} spread {:.4} mean kl {:.5} policy loss {:.4}",
            options.iterations,
            summary.mean_reward,
            summary.reward_spread,
            summary.mean_kl,
            summary.policy_loss
        ));
        running.history.push(summary);
    }

    // The last iteration describes the adapter that was written; the whole
    // history is beside it for the trend.
    let (Some(first_loss), Some(final_loss), Some(last)) = (
        running.first_loss,
        running.final_loss,
        running.history.last().cloned(),
    ) else {
        return Err(Position::unreported(resumed.as_ref(), SUBJECT));
    };
    Ok(GrpoReport {
        reward: reward.label(requested_reward),
        prompts: prompts.prompts.len(),
        trained_prompts: usable.len(),
        skipped_long: prompts.prompts.len() - usable.len(),
        group: options.group,
        turns: options.turns,
        user_model: user.map(|(_, model)| model.to_owned()),
        iterations: options.iterations,
        steps: optimizer.rates.len(),
        beta: options.beta,
        trainable_tensors: tensors,
        trainable_parameters: parameters,
        first_loss,
        final_loss,
        mean_reward: last.mean_reward,
        mean_kl: last.mean_kl,
        policy_loss: last.policy_loss,
        history: running.history,
        max_new_tokens,
        temperature: options.generation.temperature,
        top_p: options.generation.top_p,
        seed: options.generation.seed,
        rank: spec.rank,
        alpha: spec.alpha,
        targets: spec
            .targets
            .iter()
            .map(|target| target.name().to_owned())
            .collect(),
        layers: spec.layers.clone(),
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
        resumed_from: resumed.map(|position| position.step),
        checkpoints,
    })
}
