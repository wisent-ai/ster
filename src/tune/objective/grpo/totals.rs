//! The running totals of one iteration's groups, from which its report row
//! is built.

use super::GrpoIteration;
use super::group::{Group, Loss};

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
