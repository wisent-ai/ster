//! `ster workspace`: taking a pair set into the local workspace, choosing the
//! active one, taking one out, and showing what it holds.

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
    /// Make an imported pair set the active input.
    Select {
        /// Workspace name, as `ster workspace show` prints it.
        id: String,
    },
    /// Take a pair set out of the workspace and delete the workspace's copy;
    /// the source file is not touched.
    Remove {
        /// Workspace name, as `ster workspace show` prints it.
        id: String,
    },
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
        WorkspaceCommand::Select { id } => {
            super::answer(&ster::workspace::select_pair_set(&id)?)?;
        }
        WorkspaceCommand::Remove { id } => {
            super::answer(&ster::workspace::remove_pair_set(&id)?)?;
        }
    }
    Ok(())
}
