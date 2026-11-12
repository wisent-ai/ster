//! What a GRPO run refuses before it loads anything, what it keeps for its
//! report and carries in a checkpoint, and the fingerprint a checkpoint of it
//! is matched against.
//!
//! A resumed run must draw exactly the completions the stopped run would
//! have drawn next, so the draw counter that seeds every sample travels with
//! the optimizer's moments; and it must report the iteration it stopped in
//! as one iteration, so that iteration's totals travel too.

use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::super::super::preflight::Preflight;
use super::totals::Totals;
use super::{GrpoIteration, GrpoOptions};
use crate::failures::{
    refuse,
    tune::objectives::{
        GROUP_TOO_SMALL, NEW_TOKENS_NONE, POLICY_BETA_NEGATIVE, TEMPERATURE_UNUSABLE, TURNS_NONE,
        USER_MISSING, USER_UNNEEDED,
    },
};

/// How group-relative policy optimization names itself in what it says.
pub(super) const SUBJECT: &str = "group-relative policy optimization";

/// The fewest completions a group can hold: the baseline is the group's own
/// mean and spread, which one completion does not have. TRL's `GRPOConfig`
/// refuses `num_generations < 2` for the same reason:
/// https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py
const MIN_GROUP: usize = 2;

/// The loop's running report, as a checkpoint carries it.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct Running {
    pub(super) first_loss: Option<f32>,
    pub(super) final_loss: Option<f32>,
    /// Draws taken so far; the next draw's seed is advanced from it.
    pub(super) draw: u64,
    /// The totals of the iteration in progress.
    pub(super) totals: Kept,
    /// Every finished iteration, in order.
    pub(super) history: Vec<GrpoIteration>,
}

/// An iteration's totals as a checkpoint writes them.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct Kept {
    groups: usize,
    completions: usize,
    turns: usize,
    reward: f64,
    spread: f64,
    kl: f64,
    loss: f64,
    tokens: usize,
}

impl From<&Totals> for Kept {
    fn from(totals: &Totals) -> Self {
        Self {
            groups: totals.groups,
            completions: totals.completions,
            turns: totals.turns,
            reward: totals.reward,
            spread: totals.spread,
            kl: totals.kl,
            loss: totals.loss,
            tokens: totals.tokens,
        }
    }
}

impl From<&Kept> for Totals {
    fn from(kept: &Kept) -> Self {
        Self {
            groups: kept.groups,
            completions: kept.completions,
            turns: kept.turns,
            reward: kept.reward,
            spread: kept.spread,
            kl: kept.kl,
            loss: kept.loss,
            tokens: kept.tokens,
        }
    }
}

impl GrpoOptions {
    /// Refuses the options no run can train with, by name, before any model
    /// or prompt is read. `user_model` names the model given to write the
    /// user's turns, if one was.
    pub fn check(&self, user_model: Option<&str>) -> Result<()> {
        self.spec.check()?;
        self.preflight().check()?;
        if self.group < MIN_GROUP {
            return Err(refuse(
                &GROUP_TOO_SMALL,
                &[("group", &self.group.to_string())],
            ));
        }
        let turns = self.turns;
        if turns == 0 {
            return Err(refuse(
                &TURNS_NONE,
                &[
                    ("turns", &turns.to_string()),
                    ("group", &self.group.to_string()),
                ],
            ));
        }
        // A simulated user writes the turns between assistant turns, one
        // fewer than there are assistant turns.
        match (turns > 1, user_model) {
            (false, Some(model)) => {
                return Err(refuse(&USER_UNNEEDED, &[("user_model", model)]));
            }
            (true, None) => {
                return Err(refuse(&USER_MISSING, &[("turns", &turns.to_string())]));
            }
            _ => {}
        }
        // TRL's `GRPOConfig` documents `beta = 0.0` as the KL term switched
        // off, so zero is a valid beta and only a negative one is refused:
        // https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py
        if !(self.beta.is_finite() && self.beta >= 0.0) {
            return Err(refuse(
                &POLICY_BETA_NEGATIVE,
                &[("value", &self.beta.to_string())],
            ));
        }
        // Sampling divides the logits by the temperature, which Transformers'
        // `TemperatureLogitsWarper` refuses unless strictly positive:
        // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
        let temperature = self.generation.temperature;
        if !(temperature > 0.0) {
            return Err(refuse(
                &TEMPERATURE_UNUSABLE,
                &[("value", &temperature.to_string())],
            ));
        }
        if self.generation.max_new_tokens == 0 {
            return Err(refuse(
                &NEW_TOKENS_NONE,
                &[
                    ("tokens", &self.generation.max_new_tokens.to_string()),
                    ("max_sequence", &self.max_sequence.to_string()),
                ],
            ));
        }
        Ok(())
    }

    /// The shared checks, in policy optimization's words.
    pub(super) fn preflight(&self) -> Preflight<'static> {
        Preflight {
            subject: SUBJECT,
            unit: "prompt",
            pass: "iteration",
            noun: "adapter tensors",
            epochs: self.iterations,
            accumulation: self.accumulation,
            // No batch: this objective's cost is autoregressive sampling, and
            // it samples and scores one completion at a time; `--group` is
            // already the word for how many completions share a prompt.
            batch: None,
            learning_rate: self.learning_rate,
            max_sequence: self.max_sequence,
        }
    }

    /// Everything a resumed run must share with the run that wrote the
    /// checkpoint: its identity, every option, where its reward comes from
    /// and which model plays the user.
    pub(super) fn fingerprint(
        &self,
        reward: &str,
        user_model: Option<&str>,
    ) -> BTreeMap<String, Value> {
        let mut fingerprint = self.identity.clone();
        let targets: Vec<&str> = self
            .spec
            .targets
            .iter()
            .map(|target| target.name())
            .collect();
        let options = [
            ("objective", Value::from("grpo")),
            ("reward", Value::from(reward)),
            ("user_model", Value::from(user_model)),
            ("rank", Value::from(self.spec.rank)),
            ("alpha", Value::from(self.spec.alpha)),
            ("targets", Value::from(targets)),
            ("layers", Value::from(self.spec.layers.clone())),
            ("group", Value::from(self.group)),
            ("turns", Value::from(self.turns)),
            ("iterations", Value::from(self.iterations)),
            ("beta", Value::from(self.beta)),
            ("learning_rate", Value::from(self.learning_rate)),
            ("accumulation", Value::from(self.accumulation)),
            ("warmup_steps", Value::from(self.warmup_steps)),
            ("max_sequence", Value::from(self.max_sequence)),
            (
                "max_new_tokens",
                Value::from(self.generation.max_new_tokens),
            ),
            ("temperature", Value::from(self.generation.temperature)),
            ("top_p", Value::from(self.generation.top_p)),
            ("seed", Value::from(self.generation.seed)),
        ];
        fingerprint.extend(
            options
                .into_iter()
                .map(|(field, value)| (field.to_owned(), value)),
        );
        fingerprint
    }
}
