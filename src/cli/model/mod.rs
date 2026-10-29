//! Actual model operations: parser hierarchy and their shared library calls.

use std::path::PathBuf;

use anyhow::Result;
use clap::Subcommand;

use super::answer;

#[derive(Debug, Subcommand)]
pub(super) enum ModelCommand {
    /// Create native local model fixtures from an explicit declaration.
    Fixture {
        #[command(subcommand)]
        command: FixtureCommand,
    },
    /// Compile and exercise a native WordLevel tokenizer from declared tokens and pipeline.
    Tokenizer {
        #[command(subcommand)]
        command: TokenizerCommand,
    },
}

#[derive(Debug, Subcommand)]
pub(super) enum FixtureCommand {
    /// Write and load a new untrained Llama checkpoint, then forward its verification prompt.
    Create {
        /// JSON declaration naming config, tokenizer, initialization and verification inputs.
        #[arg(long = "from", value_name = "SPEC")]
        declaration: PathBuf,
        /// New checkpoint directory; missing parents are created and existing output is refused.
        #[arg(long, value_name = "DIR")]
        output: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
pub(super) enum TokenizerCommand {
    /// Write a new tokenizer document, reload it, and encode and decode the verification text.
    Create {
        /// JSON declaration of model tokens, added tokens, pipeline and verification.
        #[arg(long = "from", value_name = "SPEC")]
        declaration: PathBuf,
        /// New tokenizer JSON file; existing output is refused.
        #[arg(long, value_name = "FILE")]
        output: PathBuf,
    },
}

pub(super) fn run(command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::Fixture {
            command:
                FixtureCommand::Create {
                    declaration,
                    output,
                },
        } => answer(&ster::fixture::create(&declaration, &output)?),
        ModelCommand::Tokenizer {
            command:
                TokenizerCommand::Create {
                    declaration,
                    output,
                },
        } => answer(&ster::tokenizer::create(&declaration, &output)?),
    }
}
