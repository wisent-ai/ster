//! Writing a checkpoint: a unique attempt directory, renamed into place,
//! named in `latest`, and its predecessor removed.

use std::collections::{BTreeMap, HashMap};
use std::fs;

use anyhow::Result;
use serde_json::Value;

use super::optimizer::{AdamW, Tracked};
use super::{Durable, KIND, LATEST, Position, STATE, State, TENSORS, latest};
use crate::failures::{file, refuse, tune::checkpoint::CLOCK_UNREADABLE};

impl Durable {
    /// Write the checkpoint of `optimizer` at `position`; called only when
    /// [`Durable::due`] says one is due, which it says only with a
    /// `--checkpoint` directory.
    pub(crate) fn save(
        &self,
        optimizer: &AdamW,
        fingerprint: &BTreeMap<String, Value>,
        position: &Position,
    ) -> Result<()> {
        // Every caller runs `if options.durable.due(step)` first, and `due`
        // is `self.checkpoint.as_ref().is_some_and(..)`.
        let (directory, _) = self
            .checkpoint
            .as_ref()
            .expect("a checkpoint is due only when --checkpoint names a directory");
        let unchanged = "no checkpoint of this step was written, and latest names the checkpoint it named before";
        fs::create_dir_all(directory).map_err(|error| {
            file::write::refusal("creating", "the checkpoint directory", directory, unchanged, &error)
        })?;
        let step = position.step.to_string();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| {
                refuse(
                    &CLOCK_UNREADABLE,
                    &[("step", &step), ("error", &error.to_string())],
                )
            })?
            .as_nanos();
        let name = format!("step-{}-{}-{nonce}", position.step, std::process::id());
        let partial = directory.join(format!(".{name}.partial"));
        fs::create_dir(&partial).map_err(|error| {
            file::write::refusal("creating", "the checkpoint attempt", &partial, unchanged, &error)
        })?;
        let mut tensors = HashMap::new();
        for (tensor, Tracked { var, first, second }) in &optimizer.tracked {
            tensors.insert(format!("var/{tensor}"), var.as_tensor().clone());
            tensors.insert(format!("first/{tensor}"), first.as_tensor().clone());
            tensors.insert(format!("second/{tensor}"), second.as_tensor().clone());
        }
        let kept = format!("the unfinished attempt {} is left behind; {unchanged}", partial.display());
        let tensors_path = partial.join(TENSORS);
        candle_core::safetensors::save(&tensors, &tensors_path).map_err(|error| {
            file::tensors::write_refusal("the checkpoint's tensors", &tensors_path, &kept, &error)
        })?;
        let state = State {
            kind: KIND.to_owned(),
            written_by: format!("ster {}", env!("CARGO_PKG_VERSION")),
            fingerprint: fingerprint.clone(),
            position: position.clone(),
            rates: optimizer.rates.clone(),
        };
        let state_path = partial.join(STATE);
        let text = crate::json::pretty(&state).into_bytes();
        fs::write(&state_path, text).map_err(|error| {
            file::write::refusal("writing", "the checkpoint's state", &state_path, &kept, &error)
        })?;
        let finished = directory.join(&name);
        fs::rename(&partial, &finished).map_err(|error| {
            let action = format!("renaming {} into place as", partial.display());
            file::write::refusal(&action, "the checkpoint", &finished, &kept, &error)
        })?;
        let previous = latest(directory)?;
        let unnamed = format!(
            "checkpoint {} is written but latest still names the checkpoint before it, which --resume {} continues from",
            finished.display(),
            directory.display()
        );
        let pointer = directory.join(format!(".{name}.{LATEST}.partial"));
        fs::write(&pointer, &name).map_err(|error| {
            file::write::refusal("writing", "the draft of the latest pointer", &pointer, &unnamed, &error)
        })?;
        let named = directory.join(LATEST);
        fs::rename(&pointer, &named).map_err(|error| {
            let action = format!("renaming {} into place as", pointer.display());
            file::write::refusal(&action, "the latest pointer", &named, &unnamed, &error)
        })?;
        if let Some(previous) = previous.filter(|previous| *previous != finished) {
            fs::remove_dir_all(&previous).map_err(|error| {
                let left = format!(
                    "checkpoint {} is written and named in latest, so --resume continues from it; the one it replaced is still there",
                    finished.display()
                );
                file::write::refusal("removing", "the checkpoint latest named before", &previous, &left, &error)
            })?;
        }
        crate::workflow::progress(format!(
            "checkpoint {} written at step {}",
            finished.display(),
            position.step
        ));
        Ok(())
    }
}
