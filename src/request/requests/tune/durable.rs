//! The checkpoint fields a training operation's request takes, the same three
//! settings `ster tune <objective>` takes as `--checkpoint`,
//! `--checkpoint-every` and `--resume`.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use serde::Deserialize;

use crate::tune::Durable;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::request) struct DurableRequest {
    /// Directory the run writes its checkpoint to every `checkpointEvery`
    /// steps; the two come together or not at all.
    #[serde(default)]
    checkpoint: Option<String>,
    #[serde(default)]
    checkpoint_every: Option<NonZeroUsize>,
    /// Checkpoint directory of a stopped run to continue.
    #[serde(default)]
    resume: Option<String>,
}

impl DurableRequest {
    /// Refuses a directory without a step count, or a step count without a
    /// directory, naming the operation.
    pub(in crate::request) fn check(&self, operation: &str) -> Result<(), String> {
        match (&self.checkpoint, &self.checkpoint_every) {
            (Some(_), None) => Err(format!("{operation} names a checkpoint directory without checkpointEvery")),
            (None, Some(_)) => Err(format!("{operation} names checkpointEvery without a checkpoint directory")),
            _ => Ok(()),
        }
    }

    pub(in crate::request) fn durable(self) -> Durable {
        Durable {
            checkpoint: self.checkpoint.map(PathBuf::from).zip(self.checkpoint_every),
            resume: self.resume.map(PathBuf::from),
        }
    }
}
