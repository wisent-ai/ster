//! `ster workspace`: taking a pair set into the local workspace, choosing the
//! active one, taking one out, and showing what it holds.

use std::path::PathBuf;

use anyhow::Result;
use clap::Subcommand;

use super::onboarding;

#[derive(Debug, Subcommand)]
pub(super) enum WorkspaceCommand {
    /// Validate, persist, and activate an existing contrastive pair set.
    Import {
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
        WorkspaceCommand::Import { source, name } => {
            let mut report = ster::workspace::import_pair_set(&source, name.as_deref())?;
            let refusal = report.take_refusal();
            if refusal.is_none() {
                // An import that was not refused is imported or unchanged,
                // and both name the copy the workspace holds.
                let path = report
                    .path
                    .as_deref()
                    .expect("an accepted import names the workspace copy");
                onboarding::record_pair_set_imported(std::path::Path::new(path))?;
            }
            super::answer(&report)?;
            if let Some(refusal) = refusal {
                return Err(refusal);
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
