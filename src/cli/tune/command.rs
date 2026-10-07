//! `ster tune`'s subcommands. Each one's flags live beside its handler.

use clap::Subcommand;

use super::artifact::{EvaluateArgs, ExportArgs, InspectArgs, MergeArgs};
use super::train::{BipoArgs, DecideArgs, DpoArgs, GrpoArgs, RewardArgs, SftArgs};

#[derive(Debug, Subcommand)]
pub(in crate::cli) enum TuneCommand {
    /// Train LoRA adapters on prompt and completion examples.
    Sft(SftArgs),
    /// Train LoRA adapters to prefer one side of each contrastive pair.
    Dpo(DpoArgs),
    /// Learn one steering vector by bi-directional preference optimization
    /// (BiPO): +v prefers each pair's positive side, -v its negative side.
    Bipo(BipoArgs),
    /// Train a scalar reward head that ranks the two sides of each pair.
    Reward(RewardArgs),
    /// Optimize the policy against a reward, using a sampled group as baseline.
    Grpo(GrpoArgs),
    /// Train LoRA adapters to answer typed decisions with calibrated
    /// probabilities, from labelled decisions — Ster's RLCD.
    Decide(DecideArgs),
    /// Fold a LoRA adapter into the base weights as a standalone checkpoint.
    Merge(MergeArgs),
    /// Write a LoRA adapter as a PEFT adapter directory (adapter_config.json,
    /// adapter_model.safetensors, the base's tokenizer.json) for vLLM and peft.
    Export(ExportArgs),
    /// Score a checkpoint on held-out examples: loss and perplexity, no training.
    Evaluate(EvaluateArgs),
    /// Print and validate a Ster LoRA adapter artifact.
    Inspect(InspectArgs),
}
