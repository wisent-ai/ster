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
use serde::Serialize;

use super::{Preflight, RewardModel, Trainable, schedule, token_logprobs};
use crate::{
    model::Route,
    runtime::{Completion, DeviceChoice, GenerationOptions, Runtime},
    workflow::{self, PromptSet},
};

/// Where a completion's reward comes from.
///
/// The two arms exist for different reasons and neither is a placeholder for
/// the other. [`Reward::Length`] is a deterministic function of the completion
/// with no model behind it, which is what makes the loop runnable and checkable
/// with no judge, no artifact and no download — if reward does not rise under
/// it, the bug is in the loop. [`Reward::Model`] is the real thing.
pub enum Reward {
    Length,
    Model(Box<RewardModel>),
}

impl Reward {
    /// The keyword that selects the offline reward. A file of this name would
    /// be ambiguous; the keyword wins, and the refusal below says so.
    pub const LENGTH: &'static str = "length";

    /// Resolves `--reward`: the keyword, or a path to a reward artifact.
    pub fn parse(
        value: &str,
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
    ) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            bail!(
                "group-relative policy optimization requires a reward source; pass length or a reward artifact"
            );
        }
        if trimmed == Self::LENGTH {
            return Ok(Self::Length);
        }
        let path = Path::new(trimmed);
        if !path.exists() {
            bail!(
                "reward source {trimmed:?} is neither the keyword length nor a file that exists"
            );
        }
        workflow::progress(format!("loading the reward model at {trimmed}"));
        Ok(Self::Model(Box::new(RewardModel::load(model, revision, device, path)?)))
    }

    /// Names the source in the report, so a reward number is attributable.
    pub fn label(&self, requested: &str) -> String {
        match self {
            Self::Length => Self::LENGTH.to_owned(),
            Self::Model(_) => format!("reward:{requested}"),
        }
    }

    /// Scores one completion.
    ///
    /// The length reward counts the tokens the policy actually emitted, not the
    /// characters it decoded to: tokens are what the objective can move, and a
    /// character count would reward whichever token happens to be spelled
    /// longest. The model reward sees the whole sequence, prompt included,
    /// because a response is only good or bad relative to what it answers.
    fn score(&self, completion: &Completion) -> Result<f64> {
        match self {
            Self::Length => Ok(completion.tokens.len() as f64),
            Self::Model(model) => {
                let mut ids = completion.prompt.clone();
                ids.extend_from_slice(&completion.tokens);
                model.score(&ids)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrpoOptions {
    pub spec: crate::lora::Spec,
    /// Completions drawn per prompt. Two is the smallest group with a baseline
    /// that is not the sample itself.
    pub group: usize,
    /// Passes over the prompt set. Each one re-samples, because the point is to
    /// learn from what the *current* policy writes.
    pub iterations: usize,
    /// How hard the frozen reference pulls the policy back.
    pub beta: f64,
    pub learning_rate: f64,
    /// Prompt groups folded into one optimizer step.
    pub accumulation: usize,
    pub warmup_steps: usize,
    pub max_sequence: usize,
    pub generation: GenerationOptions,
}

/// What one pass over the prompt set produced.
#[derive(Debug, Clone, Serialize)]
pub struct GrpoIteration {
    pub iteration: usize,
    pub groups: usize,
    pub completions: usize,
    pub mean_reward: f32,
    pub reward_spread: f32,
    pub mean_kl: f32,
    pub policy_loss: f32,
    pub mean_completion_tokens: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GrpoReport {
    pub reward: String,
    pub prompts: usize,
    pub trained_prompts: usize,
    pub skipped_long: usize,
    pub group: usize,
    pub iterations: usize,
    pub steps: usize,
    pub beta: f64,
    pub trainable_tensors: usize,
    pub trainable_parameters: usize,
    pub first_loss: f32,
    pub final_loss: f32,
    /// One entry per iteration, in order. This is the shape the run is read
    /// in: a single mean over a policy that moved the whole time would hide
    /// exactly the trend the operator is looking for.
    pub history: Vec<GrpoIteration>,
    pub mean_reward: f32,
    pub mean_kl: f32,
    pub policy_loss: f32,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub layers: Vec<usize>,
    pub learning_rate: f64,
    pub accumulation: usize,
}

/// Runs the loop against `prompts`, scoring with `reward`.
///
/// `runtime` must be the one `Runtime::load_trainable` returned alongside
/// `varmap`. It plays three roles at once and can, because they differ only in
/// which forward mode they ask for: it is the policy being sampled from, the
/// policy being scored with a gradient, and — with the adapters skipped — the
/// frozen reference the KL is measured against.
pub fn grpo(
    runtime: &Runtime,
    varmap: &VarMap,
    prompts: &PromptSet,
    reward: &Reward,
    requested_reward: &str,
    options: &GrpoOptions,
) -> Result<GrpoReport> {
    if options.group < 2 {
        bail!(
            "group-relative policy optimization requires a group of at least two completions, because the group is the baseline"
        );
    }
    // The judge is a different model from the policy, but it reads the same
    // kind of residual stream, and it was fitted in whatever encoding and
    // precision trained it. A reward artifact from an f16 run scoring an f32
    // policy is reading a space it was not fitted in, and the reward it
    // returns looks perfectly ordinary, so the run says so.
    if let Some(path) = matches!(reward, Reward::Model(_)).then_some(requested_reward) {
        super::evaluate::warn_on_provenance(Path::new(path), "reward model", runtime);
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
    let Trainable { spec, vars, tensors, parameters, limit } = Preflight {
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
        ParamsAdamW { lr: options.learning_rate, ..Default::default() },
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
                let group = sample_group(runtime, prompt, reward, options, &mut draw)
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
        mean_completion_tokens: 0.0,
    });

    Ok(GrpoReport {
        reward: reward.label(requested_reward),
        prompts: prompts.prompts.len(),
        trained_prompts: usable.len(),
        skipped_long: prompts.prompts.len() - usable.len(),
        group: options.group,
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
        targets: spec.targets.iter().map(|target| target.name().to_owned()).collect(),
        layers: spec.layers.clone(),
        learning_rate: options.learning_rate,
        accumulation: options.accumulation,
    })
}

/// One sampled completion with everything the step needs about it.
struct Draw {
    completion: Completion,
    advantage: f64,
    /// The frozen reference's per-token log-probabilities of this completion.
    /// Constant: the reference cannot move, and it is scored here rather than
    /// inside the loss so the tensor carries no autograd tape.
    reference: Tensor,
}

/// A whole group for one prompt.
struct Group {
    draws: Vec<Draw>,
    mean_reward: f64,
    spread: f64,
}

/// Draws `--group` completions, scores them, and normalizes within the group.
fn sample_group(
    runtime: &Runtime,
    prompt: &str,
    reward: &Reward,
    options: &GrpoOptions,
    draw: &mut u64,
) -> Result<Group> {
    let mut completions = Vec::with_capacity(options.group);
    let mut rewards = Vec::with_capacity(options.group);
    for _ in 0..options.group {
        let generation = GenerationOptions {
            seed: options.generation.seed.wrapping_add(*draw),
            ..options.generation
        };
        *draw = draw.wrapping_add(1);
        let completion = runtime.sample(prompt, None, generation)?;
        if completion.tokens.is_empty() {
            // The first sampled token was the end of sequence. There is nothing
            // to take a gradient through, and dropping the completion would
            // bias the baseline upward by removing the group's worst member, so
            // the whole group is refused instead.
            bail!("a sampled completion was empty, so this group has nothing to score");
        }
        rewards.push(reward.score(&completion)?);
        completions.push(completion);
    }

    let count = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / count;
    let variance = rewards.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / count;
    let spread = variance.sqrt();
    // No epsilon. A group whose completions all scored the same has no
    // preference to express, and its advantages are exactly zero; adding a
    // floor to the denominator would turn that silence into amplified rounding.
    let normalize = |value: f64| if spread > 0.0 { (value - mean) / spread } else { 0.0 };

    let mut draws = Vec::with_capacity(completions.len());
    for (completion, value) in completions.into_iter().zip(&rewards) {
        let ids = sequence(&completion);
        let logits = runtime.forward_scored(&ids, Route::Base)?;
        let reference = token_logprobs(&logits, &ids, completion.prompt.len(), runtime.device())?;
        draws.push(Draw {
            advantage: normalize(*value),
            reference,
            completion,
        });
    }
    Ok(Group { draws, mean_reward: mean, spread })
}

/// The loss for one group, plus the scalars the report is built from.
struct Loss {
    tensor: Tensor,
    value: f64,
    kl: f64,
}

fn group_loss(runtime: &Runtime, group: &Group, options: &GrpoOptions) -> Result<Loss> {
    let mut summed: Option<Tensor> = None;
    let mut kl_total = 0f64;
    for draw in &group.draws {
        let ids = sequence(&draw.completion);
        let logits = runtime.forward_train(&ids)?;
        let policy = token_logprobs(&logits, &ids, draw.completion.prompt.len(), runtime.device())?;

        // pi_old is pi_theta at this exact step, so the ratio is one in value
        // and its gradient is the policy gradient. Detaching is what states
        // that, and it costs no second forward pass.
        let ratio = (&policy - policy.detach())?.exp()?;
        let advantage = (ratio * draw.advantage)?;

        // k3: exp(d) - d - 1 with d = log pi_ref - log pi_theta. Non-negative
        // for every sample and unbiased for the divergence, where the naive -d
        // is neither.
        let divergence = (&draw.reference - &policy)?;
        let penalty = ((divergence.exp()? - &divergence)? - 1.0)?;
        kl_total += penalty.mean_all()?.to_scalar::<f32>()? as f64;

        // Averaged over the completion's own tokens before the group mean, so
        // a long completion does not outvote a short one on length alone.
        let objective = (advantage - (penalty * options.beta)?)?.mean_all()?;
        let scaled = (objective.neg()? / group.draws.len() as f64)?;
        summed = Some(match summed {
            Some(total) => (total + scaled)?,
            None => scaled,
        });
    }
    let Some(tensor) = summed else {
        bail!("a sampled group contained no completions");
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
struct Totals {
    groups: usize,
    completions: usize,
    reward: f64,
    spread: f64,
    kl: f64,
    loss: f64,
    tokens: usize,
}

impl Totals {
    fn record(&mut self, group: &Group, loss: &Loss) {
        self.groups += 1;
        self.completions += group.draws.len();
        self.reward += group.mean_reward;
        self.spread += group.spread;
        self.kl += loss.kl;
        self.loss += loss.value;
        self.tokens += group.draws.iter().map(|draw| draw.completion.tokens.len()).sum::<usize>();
    }

    /// Every mean divides by the group count, guarded at one so an iteration
    /// that recorded nothing reports zero rather than a JSON `NaN` no client
    /// can parse.
    fn mean(&self, total: f64) -> f32 {
        (total / self.groups.max(1) as f64) as f32
    }

    fn mean_reward(&self) -> f32 {
        self.mean(self.reward)
    }

    fn mean_kl(&self) -> f32 {
        self.mean(self.kl)
    }

    fn finish(&self, iteration: usize) -> GrpoIteration {
        GrpoIteration {
            iteration,
            groups: self.groups,
            completions: self.completions,
            mean_reward: self.mean_reward(),
            reward_spread: self.mean(self.spread),
            mean_kl: self.mean_kl(),
            policy_loss: self.mean(self.loss),
            mean_completion_tokens: self.tokens as f32 / self.completions.max(1) as f32,
        }
    }
}
