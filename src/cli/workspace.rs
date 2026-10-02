//! `ster workspace`: taking a pair set into the local workspace, and showing
//! what it holds.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Subcommand;

use super::onboarding;

#[derive(Debug, Subcommand)]
pub(super) enum WorkspaceCommand {
    /// Validate, persist, and activate an existing contrastive pair set.
    ImportPairs {
        #[arg(value_name = "SOURCE")]
        source: PathBuf,
        /// Stable workspace name; defaults to the source file name.
        #[arg(long)]
        name: Option<String>,
    },
    /// Print imported pair sets and the active input.
    Show,
}

pub(super) fn run(command: WorkspaceCommand) -> Result<()> {
    match command {
        WorkspaceCommand::ImportPairs { source, name } => {
            let report = ster::workspace::import_pair_set(&source, name.as_deref())?;
            if report.accepted() {
                let path = report
                    .path
                    .as_deref()
                    .context("accepted pair-set import did not return a destination")?;
                onboarding::record_pair_set_imported(std::path::Path::new(path))?;
            }
            let accepted = report.accepted();
            let refusal = report.reason.clone();
            super::answer(&report)?;
            if !accepted {
                bail!(
                    "{}",
                    refusal
                        .as_deref()
                        .unwrap_or("Ster did not accept the pair set")
                );
            }
        }
        WorkspaceCommand::Show => {
            super::answer(&ster::workspace::summary()?)?;
        }
    }
    Ok(())
}
