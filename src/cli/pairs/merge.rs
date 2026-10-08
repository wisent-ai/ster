//! `ster pairs merge`: one pair set out of several, for a direction fitted
//! across all of them.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use serde_json::json;
use ster::pairs;

#[derive(Debug, Args)]
pub(super) struct MergeArgs {
    /// A pair set to merge; name two or more, in the order their pairs are
    /// written.
    #[arg(long = "pairs", required = true)]
    sources: Vec<PathBuf>,
    /// Trait name written on the merged set.
    #[arg(long = "trait")]
    trait_name: String,
    /// Pair-set JSON the merged pairs are written to.
    #[arg(long)]
    output: PathBuf,
}

pub(super) fn run(args: MergeArgs) -> Result<()> {
    let (set, report) = pairs::merge(&args.sources, &args.trait_name)?;
    set.save(&args.output)?;
    super::super::answer(&json!({
        "output": args.output.display().to_string(),
        "report": report,
    }))
}
