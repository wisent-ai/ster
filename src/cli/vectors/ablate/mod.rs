//! `ster vector ablate`: write a steering direction out of a checkpoint's
//! weights, so the model carries the change with no artifact beside it.

use std::path::PathBuf;

use anyhow::Result;

/// `ster vector ablate`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct AblateArgs {
    /// Hugging Face model id or local model directory.
    #[arg(long)]
    model: String,
    /// Immutable Hugging Face revision; defaults to main.
    #[arg(long)]
    revision: Option<String>,
    /// The steering artifact whose direction is written out of the weights.
    #[arg(long)]
    vector: PathBuf,
    /// Share of the direction removed from every residual-stream write at the
    /// artifact's layers: one removes it, a negative one amplifies it; Ster
    /// assumes none.
    #[arg(long)]
    strength: f64,
    /// Directory the rewritten checkpoint is written to.
    #[arg(long)]
    output: PathBuf,
}

pub(in crate::cli) fn ablate(args: AblateArgs) -> Result<()> {
    let AblateArgs {
        model,
        revision,
        vector,
        strength,
        output,
    } = args;
    let report = ster::tune::ablate(&model, revision.as_deref(), &vector, strength, &output)?;
    super::super::answer(&report)
}
