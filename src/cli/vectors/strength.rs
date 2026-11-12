//! `ster vector evaluate --strengths`: measuring how hard to push an artifact.

use anyhow::Result;
use ster::{
    failures::{refuse, tune::strength::SETTING_MISSING},
    tune,
};

/// The flags that turn `ster vector evaluate` into a strength selection as well.
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
    /// The strength selection these flags ask for, checked before any pair
    /// or weight is read, or nothing when no strength was named.
    pub(super) fn options(self) -> Result<Option<tune::StrengthOptions>> {
        if self.strengths.is_empty() {
            return Ok(None);
        }
        let required = |value: Option<usize>, flag: &str| {
            value.ok_or_else(|| refuse(&SETTING_MISSING, &[("flag", flag)]))
        };
        let options = tune::StrengthOptions {
            strengths: self.strengths,
            batch: required(self.batch_size, "batch-size")?,
            max_sequence: required(self.max_sequence, "max-sequence")?,
        };
        options.check()?;
        Ok(Some(options))
    }
}
