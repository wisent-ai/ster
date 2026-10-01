//! grpo.rs — group-relative policy optimization.
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

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use super::super::{
    preflight::{Preflight, Trainable},
    schedule,
};
use crate::{
    runtime::Runtime,
    workflow::{self, PromptSet},
};

mod group;
mod reward;
mod rollout;

pub use reward::{GrpoIteration, GrpoOptions, GrpoReport, Reward};
pub use rollout::UserSimulator;

use group::{Totals, group_loss, sample_group};

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
    if options.group < 2 {
        bail!(
            "group-relative policy optimization requires a group of at least two completions, because the group is the baseline"
        );
    }
    match (options.turns, user) {
        (0, _) => bail!("group-relative policy optimization requires at least one assistant turn"),
        (1, Some(_)) => bail!(
            "a simulated user writes turns between assistant turns, so it needs turns above one; this run asked for one"
        ),
        (turns, None) if turns > 1 => bail!(
            "a conversation of {turns} assistant turns needs a simulated user model to write the turns in between"
        ),
        _ => {}
    }
    let simulator = user.map(|(simulator, _)| simulator);
    // The judge is a different model from the policy, but it reads the same
    // kind of residual stream, and it was fitted in whatever encoding and
    // precision trained it. A reward artifact from an f16 run scoring an f32
    // policy is reading a space it was not fitted in, and the reward it
    // returns looks perfectly ordinary, so the run says so.
    if let Some(path) = matches!(reward, Reward::Model(_)).then_some(requested_reward) {
        crate::tune::warn_on_provenance(Path::new(path), "reward model", runtime);
    }
    if !options.beta.is_finite() || options.beta < 0.0 {
        bail!("group-relative policy optimization requires a finite beta of zero or more");
    }
    if options.generation.temperature <= 0.0 {
        bail!(
            "group-relative policy optimization requires a temperature above zero; argmax sampling would draw one identical completion per group"
        );
    }
    if options.generation.max_new_tokens == 0 {
        bail!("max_new_tokens must be greater than zero");
    }
    let Trainable {
        spec,
        vars,
        tensors,
        parameters,
        limit,
    } = Preflight {
        subject: "group-relative policy optimization",
        unit: "prompt",
        pass: "iteration",
        noun: "adapter tensors",
        epochs: options.iterations,
        accumulation: options.accumulation,
        // One sequence per forward, and no flag to say otherwise. This
        // objective's cost is autoregressive sampling, not the scoring pass,
        // and `--group` is already the word for how many completions share a
        // prompt; a second batch word here would only compete with it.
        batch: 1,
        learning_rate: options.learning_rate,
        max_sequence: options.max_sequence,
    }
    .open(runtime, varmap, &options.spec)?;

    // The prompt is tokenized once here only to decide whether the prompt plus
    // its longest possible completion can fit; the sampler tokenizes it again
    // per draw and that encode is the one the scored sequence comes from.
    let mut usable: Vec<usize> = Vec::with_capacity(prompts.prompts.len());
    for (index, prompt) in prompts.prompts.iter().enumerate() {
        let ids = runtime
            .encode(prompt)
            .with_context(|| format!("prompt {index} could not be encoded"))?;
        let longest = ids.len() + options.generation.max_new_tokens;
        if longest > limit {
            workflow::progress(format!(
                "skipping prompt {index}: {} prompt tokens plus {} sampled tokens exceed the {limit} token limit",
                ids.len(),
                options.generation.max_new_tokens
            ));
            continue;
        }
        usable.push(index);
    }
    if usable.is_empty() {
        bail!("every prompt is longer than the sequence limit, so there is nothing to train on");
    }

    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr: options.learning_rate,
            ..Default::default()
        },
    )
    .context("failed to initialize the AdamW optimizer")?;

    let steps_per_iteration = usable.len().div_ceil(options.accumulation);
    let total_steps = steps_per_iteration * options.iterations;
    let mut step = 0usize;
    let mut first_loss: Option<f32> = None;
    let mut final_loss = 0f32;
    let mut history = Vec::with_capacity(options.iterations);
    // Every draw in the whole run gets its own seed, advanced from the one the
    // operator supplied. A fixed seed would return the same completion for the
    // same prompt `--group` times over, and a group with no variety has no
    // baseline; advancing keeps the run reproducible from one number.
    let mut draw = 0u64;

    for iteration in 0..options.iterations {
        let mut totals = Totals::default();

        for group_slots in usable.chunks(options.accumulation) {
            optimizer.set_learning_rate(schedule(
                options.learning_rate,
                step,
                total_steps,
                options.warmup_steps,
            ));

            let mut summed: Option<Tensor> = None;
            let mut step_loss = 0f64;
            for &index in group_slots {
                let prompt = &prompts.prompts[index];
                workflow::progress(format!(
                    "iteration {}/{} prompt {index} sampling {} completions",
                    iteration + 1,
                    options.iterations,
                    options.group
                ));
                let group = sample_group(
                    runtime, simulator, prompt, reward, options, limit, &mut draw,
                )
                .with_context(|| format!("prompt {index} produced no usable group"))?;
                let loss = group_loss(runtime, &group, options)
                    .with_context(|| format!("prompt {index} produced no usable loss"))?;
                totals.record(&group, &loss);
                step_loss += loss.value;
                let scaled = (loss.tensor / options.accumulation as f64)?;
                summed = Some(match summed {
                    Some(total) => (total + scaled)?,
                    None => scaled,
                });
            }
            let Some(summed) = summed else {
                // `chunks` never yields an empty slice, so this is unreachable
                // in practice; refusing beats stepping on nothing.
                bail!("an accumulation group contained no prompts");
            };
            optimizer
                .backward_step(&summed)
                .context("failed to backpropagate the accumulated loss")?;

            let group_mean = (step_loss / group_slots.len() as f64) as f32;
            step += 1;
            if first_loss.is_none() {
                first_loss = Some(group_mean);
            }
            final_loss = group_mean;
            workflow::progress(format!(
                "iteration {}/{} step {step}/{total_steps} loss {group_mean:.4} reward {:.4} kl {:.5}",
                iteration + 1,
                options.iterations,
                totals.mean_reward(),
                totals.mean_kl()
            ));
        }

        let summary = totals.finish(iteration + 1);
        workflow::progress(format!(
            "iteration {}/{} mean reward {:.4} spread {:.4} mean kl {:.5} policy loss {:.4}",
            iteration + 1,
            options.iterations,
            summary.mean_reward,
            summary.reward_spread,
            summary.mean_kl,
            summary.policy_loss
        ));
        history.push(summary);
    }

    // The last iteration is the one that describes the adapter that was
    // written; the whole history is beside it for the trend.
    let last = history.last().cloned().unwrap_or(GrpoIteration {
        iteration: 0,
        groups: 0,
        completions: 0,
        mean_reward: 0.0,
        reward_spread: 0.0,
        mean_kl: 0.0,
        policy_loss: 0.0,
        mean_turns: 0.0,
        mean_completion_tokens: 0.0,
    });

    Ok(GrpoReport {
        reward: reward.label(requested_reward),
        prompts: prompts.prompts.len(),
        trained_prompts: usable.len(),
        skipped_long: prompts.prompts.len() - usable.len(),
        group: options.group,
        turns: options.turns,
        user_model: user.map(|(_, model)| model.to_owned()),
        iterations: options.iterations,
        steps: step,
        beta: options.beta,
        trainable_tensors: tensors,
        trainable_parameters: parameters,
        first_loss: first_loss.unwrap_or(final_loss),
        final_loss,
        mean_reward: last.mean_reward,
        mean_kl: last.mean_kl,
        policy_loss: last.policy_loss,
        history,
        max_new_tokens: options.generation.max_new_tokens,
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
    })
}
