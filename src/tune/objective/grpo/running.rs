//! What a GRPO run refuses before it loads anything, what it keeps for its
//! report and carries in a checkpoint, and the fingerprint a checkpoint of it
//! is matched against.
//!
//! A resumed run must draw exactly the completions the stopped run would
//! have drawn next, so the draw counter that seeds every sample travels with
//! the optimizer's moments; and it must report the iteration it stopped in
//! as one iteration, so that iteration's totals travel too.

use std::collections::BTreeMap;
use std::num::{NonZeroU32, NonZeroUsize};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::group::Totals;
use super::{GrpoIteration, GrpoOptions};

/// Whether `count` holds a second one: its base-two logarithm is above zero
/// exactly when it is two or more.
fn more_than_one(count: usize) -> bool {
    count.checked_ilog2().and_then(NonZeroU32::new).is_some()
}

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
    /// The options no run can train with, refused by name; answers the
    /// assistant turns per conversation and the token budget of one
    /// completion, both known to be above zero. `simulated_user` says
    /// whether a model was given to write the user's turns.
    pub(super) fn refusals(&self, simulated_user: bool) -> Result<(NonZeroUsize, NonZeroUsize)> {
        if !more_than_one(self.group) {
            bail!(
                "group-relative policy optimization requires a group of at least two completions, because the group is the baseline"
            );
        }
        let Some(turns) = NonZeroUsize::new(self.turns) else {
            bail!("group-relative policy optimization requires at least one assistant turn");
        };
        match (more_than_one(turns.get()), simulated_user) {
            (false, true) => bail!(
                "a simulated user writes turns between assistant turns, so it needs turns above one; this run asked for one"
            ),
            (true, false) => bail!(
                "a conversation of {turns} assistant turns needs a simulated user model to write the turns in between"
            ),
            _ => {}
        }
        if !self.beta.is_finite() || self.beta < -self.beta {
            bail!("group-relative policy optimization requires a finite beta of zero or more");
        }
        if self.generation.temperature <= -self.generation.temperature {
            bail!(
                "group-relative policy optimization requires a temperature above zero; argmax sampling would draw one identical completion per group"
            );
        }
        let Some(max_new_tokens) = NonZeroUsize::new(self.generation.max_new_tokens) else {
            bail!("max_new_tokens must be greater than zero");
        };
        Ok((turns, max_new_tokens))
    }

    /// Everything a resumed run must share with the run that wrote the
    /// checkpoint: its identity, every option, where its reward comes from
    /// and which model plays the user.
    pub(super) fn fingerprint(&self, reward: &str, user_model: Option<&str>) -> BTreeMap<String, Value> {
        let mut fingerprint = self.identity.clone();
        let targets: Vec<&str> = self.spec.targets.iter().map(|target| target.name()).collect();
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
            ("max_new_tokens", Value::from(self.generation.max_new_tokens)),
            ("temperature", Value::from(self.generation.temperature)),
            ("top_p", Value::from(self.generation.top_p)),
            ("seed", Value::from(self.generation.seed)),
        ];
        fingerprint.extend(options.into_iter().map(|(field, value)| (field.to_owned(), value)));
        fingerprint
    }
}
