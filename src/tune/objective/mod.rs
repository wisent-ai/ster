//! The four things a run can train toward. Each module owns only its own
//! loss, options and report; everything more than one of them needs lives in
//! the parent.

mod decide;
mod dpo;
mod grpo;
mod reward;
mod sft;

pub use decide::{decide, DecideOptions, DecideReport};
pub use dpo::{dpo, DpoLoss, DpoOptions, DpoReport};
pub use grpo::{grpo, GrpoIteration, GrpoOptions, GrpoReport, Reward};
pub use reward::{reward, RewardHead, RewardModel, RewardOptions, RewardReport};
pub use sft::{sft, SftOptions, SftReport};
