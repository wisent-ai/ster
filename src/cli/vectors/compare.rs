//! `ster vector compare`: how alike steering artifacts fitted for different
//! traits on one model are, and which of them group together.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ster::{
    SteeringArtifact,
    workflow::{self, CompareOptions},
};

/// `ster vector compare`
#[derive(Debug, clap::Args)]
pub(in crate::cli) struct CompareArgs {
    /// Steering artifacts to compare, two or more, fitted on one model.
    #[arg(value_name = "ARTIFACT", required = true)]
    artifacts: Vec<PathBuf>,
    /// Comma-separated layers to compare, each carried by every artifact;
    /// every layer they all carry when omitted.
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,
    /// How many groups average linkage cuts the artifacts into, at most
    /// their count; Ster assumes none.
    #[arg(long)]
    clusters: NonZeroUsize,
}

pub(in crate::cli) fn compare(args: CompareArgs) -> Result<()> {
    let CompareArgs {
        artifacts,
        layers,
        clusters,
    } = args;
    let loaded = artifacts
        .iter()
        .map(|path| {
            SteeringArtifact::load(path).with_context(|| format!("failed to compare {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let report = workflow::compare(&loaded, &CompareOptions { layers, clusters })?;
    super::super::answer(&report)
}
