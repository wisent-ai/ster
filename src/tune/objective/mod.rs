//! The four things a run can train toward. Each module owns only its own
//! loss, options and report; everything more than one of them needs lives in
//! the parent.

mod decide;
mod dpo;
mod grpo;
mod reward;
mod sft;

pub use decide::{DecideOptions, DecideReport, decide};
pub use dpo::{BipoOptions, BipoReport, DpoLoss, DpoOptions, DpoReport, bipo, dpo};
pub use grpo::{
    GrpoIteration, GrpoOptions, GrpoReport, Reward, Rollout, UserSimulator, grpo, rollout,
};
pub use reward::{RewardHead, RewardModel, RewardOptions, RewardReport, reward};
pub use sft::{SftOptions, SftReport, sft};
