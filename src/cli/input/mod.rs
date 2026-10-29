//! Native requests deserialize the actual CLI arguments, not a second schema.

use std::path::Path;

use ster::{ChatChoice, DeviceChoice, Precision};

use super::ModelArgs;

pub(crate) trait InputCheck {
    fn check(&self) -> Result<(), String>;
}

pub(super) fn required_text(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} requires a nonempty value, got {value:?}"));
    }
    Ok(())
}

pub(super) fn required_path(value: &Path, field: &str) -> Result<(), String> {
    if value.as_os_str().is_empty() || value.to_str().is_some_and(|text| text.trim().is_empty()) {
        return Err(format!("{field} requires a nonempty path, got {value:?}"));
    }
    Ok(())
}

impl InputCheck for ModelArgs {
    fn check(&self) -> Result<(), String> {
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
