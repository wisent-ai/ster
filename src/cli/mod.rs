//! Everything the `ster` binary itself owns: the command surface a person
//! types, and the arms behind it. The library does the work; this decides
//! what was asked for and prints the answer.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser};
use ster::{DeviceChoice, Precision, Runtime};

mod command;
mod decide;
mod onboarding;
mod pairs;
mod tune;
mod vectors;
mod workspace;

use command::Command;

#[derive(Debug, Parser)]
#[command(
    name = "ster",
    version,
    about = "Understand, measure, and control model representations",
    long_about = "Ster reads hidden representations from open-weight Llama-family models, trains steering directions from contrastive pairs, evaluates those directions, and applies them during generation."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Args)]
pub(crate) struct ModelArgs {
    /// Hugging Face model id or local model directory.
    #[arg(long)]
    model: String,
    /// Immutable Hugging Face revision; defaults to main.
    #[arg(long)]
    revision: Option<String>,
    /// Runtime device: cpu, metal, or cuda.
    #[arg(long, default_value = "cpu")]
    device: String,
}

impl ModelArgs {
    /// The shared load. Every command that maps a checkpoint goes through
    /// here, so `--precision` means the same thing on all of them and a new
    /// command cannot quietly forget it.
    fn load_at(&self, precision: Precision) -> Result<Runtime> {
        let device = DeviceChoice::parse(&self.device)?;
        Runtime::load_at(&self.model, self.revision.as_deref(), device, precision)
    }
}

pub(crate) fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Train(args) => vectors::train(args),
        Command::Optimize(args) => vectors::optimize(args),
        Command::Evaluate(args) => vectors::evaluate(args),
        Command::Generate(args) => vectors::generate(args),
        Command::Extract(args) => vectors::extract(args),
        Command::Inspect(args) => vectors::inspect(args),
        Command::Onboarding(args) => vectors::onboarding(args),
        Command::Decide(args) => decide::decide(args),
        Command::Calibrate(args) => decide::calibrate(args),
        Command::Workspace { command } => workspace::run(command),
        Command::Pairs { command } => pairs::run(command),
        Command::Tune { command } => tune::run(command),
        Command::Serve { port } => ster::serve::run(port),
    }
}

fn resolve_pairs(selected: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = selected {
        return Ok(path);
    }
    ster::workspace::active_pair_set()?.context(
        "no active Ster pair set; pass --pairs or import one with `ster workspace import-pairs`",
    )
}
