//! The two flag parsers every adapter-training request shares.

use anyhow::{bail, Result};

use crate::{lora, workflow::parse_layers};

/// `targets` is a comma-separated projection list, exactly as `--targets` is
/// on the CLI. Repeats collapse and the order follows the request.
pub(super) fn parse_targets(value: &str) -> Result<Vec<lora::Target>> {
    let mut targets = Vec::new();
    for segment in value
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        let target = lora::Target::parse(segment)?;
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    if targets.is_empty() {
        bail!("no targets selected");
    }
    Ok(targets)
}

/// `layers` means what it means everywhere else in Ster, with one difference:
/// `all` cannot be expanded yet. `parse_layers` needs the model's layer count,
/// and the count is only known once the weights are mapped — which happens
/// inside `Runtime::load_trainable`, after the spec exists. An empty layer
/// list is the spec's way of saying every layer, and the loader resolves it
/// against the real count before it builds any adapter.
pub(super) fn parse_adapter_layers(value: &str) -> Result<Vec<usize>> {
    if value.trim() == "all" {
        return Ok(Vec::new());
    }
    parse_layers(value, usize::MAX)
}
