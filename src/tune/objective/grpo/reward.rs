//! Where a completion's reward comes from, and what a run is asked for.

use std::path::Path;

use anyhow::{Result, bail};
use serde::Serialize;

use crate::{
    runtime::{Completion, DeviceChoice, GenerationOptions},
    workflow,
};

use super::super::RewardModel;
use super::scorer::Scorer;

/// Where a completion's reward comes from.
///
/// The arms exist for different reasons and none is a placeholder for
/// another. [`Reward::Length`] is a deterministic function of the completion
/// with no model behind it, which is what makes the loop runnable and checkable
/// with no judge, no artifact and no download — if reward does not rise under
/// it, the bug is in the loop. [`Reward::Model`] is a reward Ster trained;
/// [`Reward::Scorer`] is one only an outside service can give.
pub enum Reward {
    Length,
    Model(Box<RewardModel>),
    /// An outside scorer over HTTP ([`Scorer`]); the completion's text only.
    Scorer(Scorer),
}

impl Reward {
    /// The keyword that selects the offline reward. A file of this name would
    /// be ambiguous; the keyword wins, and the refusal below says so.
    pub const LENGTH: &'static str = "length";

    /// Resolves `--reward`: the keyword, an `http(s)://…#/pointer` scorer,
    /// or a path to a reward artifact.
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
        if let Some(scorer) = Scorer::parse(trimmed)? {
            return Ok(Self::Scorer(scorer));
        }
        let path = Path::new(trimmed);
        if !path.exists() {
            bail!("reward source {trimmed:?} is neither the keyword length nor a file that exists");
        }
        workflow::progress(format!("loading the reward model at {trimmed}"));
        Ok(Self::Model(Box::new(RewardModel::load(
            model, revision, device, path,
        )?)))
    }

    /// Names the source in the report, so a reward number is attributable.
    pub fn label(&self, requested: &str) -> String {
        match self {
            Self::Length => Self::LENGTH.to_owned(),
            Self::Model(_) => format!("reward:{requested}"),
            Self::Scorer(_) => format!("scorer:{requested}"),
        }
    }

    /// Scores one completion.
    ///
    /// The length reward counts the tokens the policy actually emitted, not the
    /// characters it decoded to: tokens are what the objective can move, and a
    /// character count would reward whichever token happens to be spelled
    /// longest. The model reward sees the whole sequence, prompt included,
    /// because a response is only good or bad relative to what it answers.
    pub(super) fn score(&self, completion: &Completion) -> Result<f64> {
        match self {
            Self::Length => Ok(completion.tokens.len() as f64),
            Self::Model(model) => {
                let mut ids = completion.prompt.clone();
                ids.extend_from_slice(&completion.tokens);
                model.score(&ids)
            }
            Self::Scorer(scorer) => scorer.score(&completion.text),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrpoOptions {
    pub spec: crate::lora::Spec,
    /// Completions drawn per prompt. Two is the smallest group with a baseline
    /// that is not the sample itself.
    pub group: usize,
    /// Assistant turns per conversation. One is a single completion per
    /// prompt; above one, a simulated user writes the turns in between.
    pub turns: usize,
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
    /// Assistant turns per conversation, averaged: below `turns` when the
    /// simulated user ended conversations early.
    pub mean_turns: f32,
    pub mean_completion_tokens: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct GrpoReport {
    pub reward: String,
    pub prompts: usize,
    pub trained_prompts: usize,
    pub skipped_long: usize,
    pub group: usize,
    pub turns: usize,
    /// The model that wrote the user's turns, when there were any.
    pub user_model: Option<String>,
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
