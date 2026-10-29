//! Typed training arguments and their shared operation handlers.
//! Handlers return the actual document; CLI and request frontends render it.

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

fn finish(
    artifact: ster::lora::Artifact,
    output: &std::path::Path,
) -> anyhow::Result<serde_json::Value> {
    artifact.save(output)?;
    Ok(serde_json::Value::Object(serde_json::Map::from_iter([
        (
            "path".to_owned(),
            serde_json::Value::String(output.display().to_string()),
        ),
        ("report".to_owned(), artifact.train),
    ])))
}
