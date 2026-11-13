//! The input formats a trainer reads, and how a run turns them into the
//! padded rows one forward pass takes.

use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use crate::failures::{
    file, refuse,
    tune::examples::{BLANK, EMPTY, FORM, MISSING, SYSTEM_BLANK},
};

/// What an example set is called in a refusal of its file.
const ROLE: &str = "the example set";

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
        let shown = path.display().to_string();
        let bytes = std::fs::read(path)
            .map_err(|error| file::read::refusal_or(&MISSING, ROLE, path, &error))?;
        // `{"examples": [...]}` is Ster's own set; a list or JSON Lines of rows is
        // a set another trainer publishes (`imported.rs`).
        let rows = match serde_json::from_slice::<serde_json::Value>(&bytes) {
            Ok(serde_json::Value::Object(set)) if set.contains_key("examples") => {
                // Read from the bytes, so a refusal names the line and column it stopped at.
                let value: Self = serde_json::from_slice(&bytes)
                    .map_err(|error| file::json::refusal(ROLE, FORM, path, &error))?;
                value.validate(&shown)?;
                return Ok(value);
            }
            Ok(serde_json::Value::Array(rows)) => rows,
            Ok(row) => vec![row],
            Err(error) => super::imported::lines(&bytes).ok_or_else(|| file::json::refusal(ROLE, FORM, path, &error))?,
        };
        let value = Self { name: String::new(), examples: super::imported::examples(&rows, &shown)? };
        value.validate(&shown)?;
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
            return Err(refuse(&EMPTY, &[("label", label), ("name", &self.name)]));
        }
        let blank = |text: &str| text.trim().is_empty();
        if let Some((index, example)) = self
            .examples
            .iter()
            .enumerate()
            .find(|(_, example)| blank(&example.prompt) || blank(&example.completion))
        {
            return Err(refuse(
                &BLANK,
                &[
                    ("label", label),
                    ("index", &index.to_string()),
                    ("prompt", &format!("{:?}", example.prompt)),
                    ("completion", &format!("{:?}", example.completion)),
                ],
            ));
        }
        if let Some((index, system)) = self.examples.iter().enumerate().find_map(|(index, example)| {
            example
                .system
                .as_deref()
                .filter(|system| blank(system))
                .map(|system| (index, system))
        }) {
            return Err(refuse(
                &SYSTEM_BLANK,
                &[
                    ("label", label),
                    ("index", &index.to_string()),
                    ("system", &format!("{system:?}")),
                ],
            ));
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
