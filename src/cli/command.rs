//! Every top-level `ster` command and the flags it takes. The bodies live
//! beside the surfaces they drive; this is the contract a person types.

use clap::Subcommand;

use super::converse::ConverseArgs;
use super::decide::{CalibrateArgs, DecideArgs};
use super::decisions::DecisionsCommand;
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
    /// Play conversations between a model and a simulated user and write
    /// each as {"messages": [...]}: the rollout tune grpo --turns trains on,
    /// with no gradient.
    Converse(ConverseArgs),
    /// Import existing contrastive data during first use, or replay the walkthrough.
    Onboarding(OnboardingArgs),
    /// Answer typed questions about a state from one forward pass, with a
    /// probability for every option.
    Decide(DecideArgs),
    /// Fit the temperature that makes decision probabilities honest, from
    /// labelled examples.
    Calibrate(CalibrateArgs),
    /// Get, split, and benchmark labelled decisions: the pipeline around a
    /// decision model.
    Decisions {
        #[command(subcommand)]
        command: DecisionsCommand,
    },
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
    /// Write a tiny offline Llama-family checkpoint (seeded random weights,
    /// a ~60-word WordLevel tokenizer) that every command can load with no
    /// download, GPU or account.
    ToyModel {
        #[arg(value_name = "DIR")]
        out: std::path::PathBuf,
    },
    /// Run one JSON request from a desktop app to completion: the body on
    /// stdin, NDJSON log events and one result event on stdout, and the
    /// result's status as the exit status.
    Request {
        /// The operation: train, optimize, evaluate, generate, extract,
        /// inspect, decide, calibrate, workspace/import-pairs, workspace/show,
        /// workspace/select, workspace/remove, pairs/inspect, pairs/save,
        /// pairs/synthesize, or tune/sft, tune/dpo, tune/reward, tune/grpo,
        /// tune/merge, tune/evaluate, tune/inspect.
        operation: String,
    },
}
