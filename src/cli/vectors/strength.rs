//! `ster evaluate --strengths`: measuring how hard to push an artifact.

use anyhow::{Context, Result};
use ster::{PairSet, Runtime, SteeringArtifact, tune};

/// The flags that turn `ster evaluate` into a strength selection as well.
#[derive(Debug, clap::Args)]
pub(super) struct StrengthArgs {
    /// Strengths to measure the artifact at, comma-separated. With them,
    /// evaluate also scores how each one moves the model's log-probability of
    /// every pair's positive side against its negative side, and picks one.
    #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
    strengths: Vec<f64>,
    /// Pairs scored in one forward pass while measuring strengths; required
    /// with --strengths.
    #[arg(long, requires = "strengths")]
    batch_size: Option<usize>,
    /// The longest pair side, in tokens, that is measured; longer pairs are
    /// skipped and counted. Required with --strengths.
    #[arg(long, requires = "strengths")]
    max_sequence: Option<usize>,
}

impl StrengthArgs {
    /// The strength selection these flags ask for, or nothing when no
    /// strength was named.
    pub(super) fn measure(
        self,
        runtime: &Runtime,
        pairs: &PairSet,
        artifact: &SteeringArtifact,
    ) -> Result<Option<tune::StrengthReport>> {
        if self.strengths.is_empty() {
            return Ok(None);
        }
        let options = tune::StrengthOptions {
            strengths: self.strengths,
            batch: self
                .batch_size
                .context("--strengths needs --batch-size; Ster assumes none")?,
            max_sequence: self
                .max_sequence
                .context("--strengths needs --max-sequence; Ster assumes none")?,
        };
        tune::strengths(runtime, pairs, artifact, &options).map(Some)
    }
}
