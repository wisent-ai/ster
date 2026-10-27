//! The flags every `ster tune train` objective takes to keep its state
//! during a run and to continue a run that stopped.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use ster::tune;

#[derive(Debug, clap::Args)]
pub(in crate::cli) struct DurableArgs {
    /// Directory the run writes its checkpoint to every --checkpoint-every
    /// steps: the adapter, the optimizer's moments and steps, and where the
    /// run stands. Only the newest checkpoint is kept.
    #[arg(long, requires = "checkpoint_every")]
    checkpoint: Option<PathBuf>,
    /// Steps between two checkpoints; Ster assumes none.
    #[arg(long, requires = "checkpoint")]
    checkpoint_every: Option<NonZeroUsize>,
    /// Checkpoint directory of a stopped run to continue: the run started now
    /// must be the same run (model, revision, data, options, seed), and is
    /// refused naming every field that differs.
    #[arg(long)]
    resume: Option<PathBuf>,
}

impl DurableArgs {
    pub(in crate::cli) fn durable(self) -> tune::Durable {
        tune::Durable {
            checkpoint: self.checkpoint.zip(self.checkpoint_every),
            resume: self.resume,
        }
    }
}
