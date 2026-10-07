//! Exporting hidden representations for a plain list of prompts, and the
//! document that file format is.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::runtime::Runtime;

use super::progress;

pub fn extract(runtime: &Runtime, input: &Path, output: &Path, layers: &[usize]) -> Result<()> {
    let prompts = PromptSet::load(input)?;
    let mut records = Vec::with_capacity(prompts.prompts.len());
    for (index, prompt) in prompts.prompts.iter().enumerate() {
        progress(format!(
            "extracting prompt {}/{}",
            index + 1,
            prompts.prompts.len()
        ));
        let activations = runtime.activations(prompt, layers)?;
        records.push(ActivationRecord {
            prompt: prompt.clone(),
            layers: activations.into_iter().collect(),
        });
    }
    let artifact = ActivationArtifact {
        schema_version: 1,
        product: "ster".to_owned(),
        model: runtime.model_id.clone(),
        model_revision: runtime.revision.clone(),
        hidden_size: runtime.hidden_size(),
        records,
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec(&artifact)?)
        .with_context(|| format!("failed to write {}", output.display()))
}

/// A bare list of prompts: `{"prompts": ["…"]}`.
///
/// `extract` defined this shape and policy optimization reads the same file,
/// so it is one public type rather than two structs that agree by accident.
/// A prompt set states what to ask; what to do with the answers is the
/// command's business.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptSet {
    pub prompts: Vec<String>,
}

impl PromptSet {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid prompt JSON in {}", path.display()))?;
        if value.prompts.is_empty() {
            bail!("prompt set contains no prompts");
        }
        Ok(value)
    }
}

#[derive(Debug, Serialize)]
struct ActivationArtifact {
    schema_version: u32,
    product: String,
    model: String,
    model_revision: Option<String>,
    hidden_size: usize,
    records: Vec<ActivationRecord>,
}

#[derive(Debug, Serialize)]
struct ActivationRecord {
    prompt: String,
    layers: BTreeMap<usize, Vec<f32>>,
}
