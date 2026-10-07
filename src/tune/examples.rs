//! The input formats a trainer reads, and how a run turns them into the
//! padded rows one forward pass takes.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

// MARK: - Example sets

#[derive(Debug, Clone, Deserialize)]
pub struct Example {
    /// The system turn the model is served with, when it is served with one.
    /// Only a chat template knows where a system turn goes, so an example
    /// carrying one is refused on a run whose template is absent or off.
    #[serde(default)]
    pub system: Option<String>,
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExampleSet {
    #[serde(default)]
    pub name: String,
    pub examples: Vec<Example>,
}

impl ExampleSet {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read example set {}", path.display()))?;
        let value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid example set JSON in {}", path.display()))?;
        value.validate(&path.display().to_string())?;
        Ok(value)
    }

    /// Checks the two content invariants every consumer of an example set
    /// relies on.
    ///
    /// `label` is the identity quoted in the refusal sentence, exactly as
    /// `PairSet::validate` uses it: the loader passes the path, and a caller
    /// validating a set assembled from an API request passes whatever names it
    /// to the operator. Both sentences are published in the runbook, so they
    /// must stay byte-identical regardless of which caller triggers them.
    pub fn validate(&self, label: &str) -> Result<()> {
        if self.examples.is_empty() {
            bail!("example set {label} contains no examples");
        }
        if self
            .examples
            .iter()
            .any(|example| example.prompt.trim().is_empty() || example.completion.trim().is_empty())
        {
            bail!("example set {label} contains an empty prompt or completion");
        }
        if self
            .examples
            .iter()
            .any(|example| example.system.as_deref().is_some_and(|system| system.trim().is_empty()))
        {
            bail!("example set {label} contains an empty system prompt");
        }
        Ok(())
    }

    /// How the set names itself in a refusal when it never came from a file.
    pub(crate) fn label(&self) -> String {
        match self.name.trim() {
            "" => "(unnamed)".to_owned(),
            name => name.to_owned(),
        }
    }
}
