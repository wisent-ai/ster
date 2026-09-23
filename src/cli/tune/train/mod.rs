//! The four training arms of `ster tune`. Each objective owns its own flags
//! and its own arm; all of them go through the same `tune` and `lora`
//! functions `ster request` calls and print one pretty JSON document.

mod decide;
mod dpo;
mod grpo;
mod reward;
mod sft;

pub(super) use decide::{decide, DecideArgs};
pub(super) use dpo::{dpo, DpoArgs};
pub(super) use grpo::{grpo, GrpoArgs};
pub(super) use reward::{reward, RewardArgs};
pub(super) use sft::{sft, SftArgs};
