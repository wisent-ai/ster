//! `ster parity`: comparing this model's forward pass with hidden states
//! another implementation recorded for the same token ids.

use std::path::PathBuf;

use anyhow::Result;
use ster::{Precision, workflow};

use super::super::{ModelArgs, answer};

/// `ster parity`
#[derive(Debug, clap::Args)]
pub(in super::super) struct ParityArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// JSON file shaped as {"records": [{"tokenIds": [...] or "tokenIdsFile":
    /// "ids.json", "hidden": "states.safetensors", "tensor": "hidden_states"}]}:
    /// token ids and the states after the final norm another implementation
    /// recorded for their first positions, [positions, hidden]; paths are
    /// relative to this file and "tensor" defaults to hidden_states. A
    /// reference holding no positions is refused; meanCosine and
    /// smallestCosine are null when no compared position has a nonzero state
    /// on both sides.
    #[arg(long)]
    input: PathBuf,
    /// Dtype the base weights are mapped at: f32, f16, or bf16. The states
    /// are compared in F32 either way; this is the width Ster computed them
    /// in. bf16 needs --device metal or cuda.
    #[arg(long, default_value = "f32", value_parser = Precision::parse)]
    precision: Precision,
}

pub(in super::super) fn parity(args: ParityArgs) -> Result<()> {
    let ParityArgs {
        model,
        input,
        precision,
    } = args;
    let runtime = model.load_at(precision)?;
    answer(&workflow::parity(&runtime, &input)?)?;
    Ok(())
}
