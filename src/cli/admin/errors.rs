//! `ster errors`: the failure catalogue every functionality declares in
//! `ster::failures`, listed, shown one functionality at a time, and checked.

use std::path::PathBuf;

use anyhow::Result;
use clap::{CommandFactory, Subcommand};

#[derive(Debug, Subcommand)]
pub(super) enum ErrorsCommand {
    /// Print every functionality: the entries that start it, what it uses,
    /// and every cause it declares with its code, exit status, test and
    /// documentation anchor.
    List {
        /// Print the catalogue's documentation sections instead of JSON.
        #[arg(long)]
        markdown: bool,
        /// With --markdown, only the functionalities listed on this page, e.g.
        /// docs/guide/errors/workspace.md: what it holds below its marker.
        #[arg(long, requires = "markdown")]
        page: Option<String>,
    },
    /// Print one functionality's causes and every cause it meets through the
    /// functionalities it uses.
    Show {
        /// The functionality's id, as `ster errors list` prints it, e.g.
        /// ster.workspace.import.
        functionality: String,
    },
    /// Check every declaration, every declared test this checkout holds, and
    /// every command no functionality names; exits 1 on any finding.
    Check {
        /// The checkout whose tests the declared causes name.
        #[arg(long)]
        repository: Option<PathBuf>,
        /// A JSON list of the commands still allowed to be uncatalogued, e.g.
        /// tests/errors/uncatalogued.json; it may only shrink.
        #[arg(long)]
        uncatalogued: Option<PathBuf>,
        /// A JSON list of the declared tests not yet written, e.g.
        /// tests/errors/untested.json; it may only shrink.
        #[arg(long)]
        untested: Option<PathBuf>,
    },
}

pub(super) fn run(command: ErrorsCommand) -> Result<()> {
    match command {
        ErrorsCommand::List { markdown, page } => {
            if markdown {
                print!("{}", ster::failures::markdown(page.as_deref())?);
            } else {
                super::answer_document(&ster::failures::list()?)?;
            }
        }
        ErrorsCommand::Show { functionality } => {
            super::answer_document(&ster::failures::show(&functionality)?)?;
        }
        ErrorsCommand::Check {
            repository,
            uncatalogued,
            untested,
        } => {
            let answer = ster::failures::check(
                repository.as_deref(),
                uncatalogued.as_deref(),
                untested.as_deref(),
                &commands(),
            )?;
            super::answer_document(&answer)?;
            if let Some(refusal) = ster::failures::findings(&answer) {
                return Err(refusal);
            }
        }
    }
    Ok(())
}

/// Every command `ster` answers, by its path below `ster`: `workspace
/// import`, `generate`.
fn commands() -> Vec<String> {
    let mut paths = Vec::new();
    leaves(&super::Cli::command(), "", &mut paths);
    paths
}

fn leaves(command: &clap::Command, prefix: &str, paths: &mut Vec<String>) {
    for subcommand in command.get_subcommands() {
        let name = subcommand.get_name();
        let path = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix} {name}")
        };
        if subcommand.has_subcommands() {
            leaves(subcommand, &path, paths);
        } else {
            paths.push(path);
        }
    }
}
