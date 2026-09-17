//! Every top-level `ster` command and the flags it takes. The bodies live
//! beside the surfaces they drive; this is the contract a person types.

use clap::Subcommand;

use super::decide::{CalibrateArgs, DecideArgs};
use super::pairs::PairsCommand;
use super::tune::TuneCommand;
use super::vectors::{
    EvaluateArgs, ExtractArgs, GenerateArgs, InspectArgs, OnboardingArgs, OptimizeArgs, TrainArgs,
};
use super::workspace::WorkspaceCommand;

#[derive(Debug, Subcommand)]
pub(super) enum Command {
    /// Train steering vectors from positive and negative prompts.
    Train(TrainArgs),
    /// Select the best method and layer on an 80/20 holdout.
    Optimize(OptimizeArgs),
    /// Measure pair ordering for a steering artifact.
    Evaluate(EvaluateArgs),
    /// Generate text with an optional steering artifact.
    Generate(GenerateArgs),
    /// Export hidden representations for arbitrary prompts.
    Extract(ExtractArgs),
    /// Summarize and validate a Ster steering artifact.
    Inspect(InspectArgs),
    /// Import existing contrastive data during first use, or replay the walkthrough.
    Onboarding(OnboardingArgs),
    /// Answer typed questions about a state from one forward pass, with a
    /// probability for every option.
    Decide(DecideArgs),
    /// Fit the temperature that makes decision probabilities honest, from
    /// labelled examples.
    Calibrate(CalibrateArgs),
    /// Import and inspect Ster's persistent local workspace.
    Workspace {
        #[command(subcommand)]
        command: WorkspaceCommand,
    },
    /// Author, inspect, and synthesize contrastive pair sets.
    Pairs {
        #[command(subcommand)]
        command: PairsCommand,
    },
    /// Train, merge, score, and inspect LoRA adapters.
    Tune {
        #[command(subcommand)]
        command: TuneCommand,
    },
    /// Loopback HTTP/JSON backend for desktop apps.
    Serve {
        /// Port to bind; 0 selects an ephemeral port.
        #[arg(long, default_value_t = 0)]
        port: u16,
    },
}
