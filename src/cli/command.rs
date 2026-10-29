//! Every top-level `ster` command and the flags it takes. The bodies live
//! beside the surfaces they drive; this is the contract a person types.

use clap::Subcommand;

use super::converse::ConverseArgs;
use super::decide::{CalibrateArgs, DecideArgs};
use super::decisions::DecisionsCommand;
use super::model::ModelCommand;
use super::pairs::PairsCommand;
use super::tune::TuneCommand;
use super::vectors::{ExtractArgs, GenerateArgs, OnboardingArgs, ParityArgs, VectorCommand};
use super::workspace::WorkspaceCommand;

#[derive(Debug, Subcommand)]
pub(super) enum Command {
    /// Train, select, score, read, and compare steering vectors: the
    /// directions every steered run adds to the residual stream.
    Vector {
        #[command(subcommand)]
        command: VectorCommand,
    },
    /// Generate text with an optional steering artifact.
    Generate(GenerateArgs),
    /// Export hidden representations for arbitrary prompts.
    Extract(ExtractArgs),
    /// Compare this model's states after the final norm with the ones
    /// another implementation recorded for the same token ids.
    Parity(ParityArgs),
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
    /// Create and verify native local fixtures and tokenizers from explicit inputs.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Run one JSON request from a desktop app to completion: the body on
    /// stdin, NDJSON log events and one result event on stdout, and the
    /// result's status as the exit status.
    Request {
        /// The operation: vector/train, vector/optimize, vector/curve, vector/evaluate,
        /// vector/inspect, vector/compare, vector/project, vector/ablate, generate, extract, parity, decide,
        /// calibrate, model/fixture/create, model/tokenizer/create, workspace/import, workspace/show,
        /// workspace/select, workspace/remove, pairs/import, pairs/inspect, pairs/merge,
        /// pairs/save, pairs/synthesize, or tune/sft, tune/dpo, tune/reward,
        /// tune/grpo, tune/merge, tune/evaluate, tune/inspect.
        operation: String,
    },
}
