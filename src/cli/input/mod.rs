//! Native requests deserialize the actual CLI arguments, not a second schema.

use std::path::Path;

use anyhow::Result;
use ster::{ChatChoice, DeviceChoice, Precision, failures::request::require};

use super::ModelArgs;

pub(crate) trait InputCheck {
    fn check(&self) -> Result<()>;
}

/// The operation, request field and CLI flag a check's `label` names:
/// `"tune/sft examples"` is field `examples` of `tune/sft`, read by
/// `--examples`; a bare `"model"` is a field every tuning operation shares.
fn named(label: &str) -> (&str, &str, String) {
    let (operation, field) = match label.rsplit_once(' ') {
        Some(parts) => parts,
        None => ("tune", label),
    };
    let mut flag = String::from("--");
    for character in field.chars() {
        if character.is_ascii_uppercase() {
            flag.push('-');
            flag.push(character.to_ascii_lowercase());
        } else {
            flag.push(character);
        }
    }
    (operation, field, flag)
}

pub(super) fn required_text(value: &str, label: &str) -> Result<()> {
    let (operation, field, flag) = named(label);
    require(value, operation, field, "a nonempty value", &flag)
}

pub(super) fn required_path(value: &Path, label: &str) -> Result<()> {
    let (operation, field, flag) = named(label);
    require(&value.to_string_lossy(), operation, field, "a nonempty path", &flag)
}

impl InputCheck for ModelArgs {
    fn check(&self) -> Result<()> {
        required_text(&self.model, "model")?;
        if let Some(revision) = &self.revision {
            required_text(revision, "revision")?;
        }
        Ok(())
    }
}

// Categorical defaults are the existing declared CLI grammar. These choices
// remain protocol identity, not removed classifiers. No experiment number
// defaults here: omission must be reported by the actual argument schema.
pub(super) fn default_device() -> DeviceChoice {
    DeviceChoice::Cpu
}
pub(super) fn default_precision() -> Precision {
    Precision::F32
}
pub(super) fn default_chat_template() -> ChatChoice {
    ChatChoice::Auto
}

pub(super) const TARGETS: &str = "query,value";
pub(super) const LAYERS: &str = "all";
pub(super) const LOSS: &str = "dpo";
pub(super) const REWARD: &str = "length";

pub(super) fn default_targets() -> String {
    TARGETS.to_owned()
}
pub(super) fn default_layers() -> String {
    LAYERS.to_owned()
}
pub(super) fn default_loss() -> String {
    LOSS.to_owned()
}
pub(super) fn default_reward() -> String {
    REWARD.to_owned()
}
