//! The training arms of `ster tune`. Each objective owns its own flags and
//! its own arm; all of them go through the same `tune` and `lora` functions
//! `ster request` calls and print one pretty JSON document.

mod bipo;
mod decide;
mod dpo;
mod grpo;
mod reward;
mod sft;

pub(super) use bipo::{BipoArgs, bipo};
pub(super) use decide::{DecideArgs, decide};
pub(super) use dpo::{DpoArgs, dpo};
pub(super) use grpo::{GrpoArgs, grpo};
pub(super) use reward::{RewardArgs, reward};
pub(super) use sft::{SftArgs, sft};
