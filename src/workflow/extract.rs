//! Exporting hidden representations for a plain list of prompts, and the
//! document that file format is.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::failures::{file, workflow::prompts as causes};
use crate::runtime::Runtime;

use super::progress;

/// What the written file is, for the `ster.file` refusals that name it.
const OUTPUT: &str = "the activations ster extract writes";

pub fn extract(
    runtime: &Runtime,
    prompts: &PromptSet,
    output: &Path,
    layers: &[usize],
) -> Result<()> {
    let mut records = Vec::with_capacity(prompts.prompts.len());
    for (index, prompt) in prompts.prompts.iter().enumerate() {
        progress(format!(
            "extracting prompt {}/{}",
            index + 1,
            prompts.prompts.len()
        ));
        let activations = runtime.activations(prompt, layers)?;
        for (layer, values) in &activations {
            if let Some((position, value)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(crate::failures::refuse(
                    &causes::STATES_NOT_FINITE,
                    &[
                        ("index", &index.to_string()),
                        ("prompts", &prompts.prompts.len().to_string()),
                        ("layer", &layer.to_string()),
                        ("position", &position.to_string()),
                        ("width", &values.len().to_string()),
                        ("value", &value.to_string()),
                        ("precision", runtime.precision().name()),
                        ("output", &output.display().to_string()),
                    ],
                ));
            }
        }
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
    let left = format!(
        "nothing was written; the {} prompts were read and are not kept",
        prompts.prompts.len()
    );
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            file::write::refusal("creating the directory of", OUTPUT, output, &left, &error)
        })?;
    }
    fs::write(output, crate::json::compact(&artifact).into_bytes())
        .map_err(|error| file::write::refusal("writing", OUTPUT, output, &left, &error))
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
        let role = "the prompt set";
        let bytes = fs::read(path)
            .map_err(|error| file::read::refusal_or(&causes::MISSING, role, path, &error))?;
        let value: Self = serde_json::from_slice(&bytes)
            .map_err(|error| file::json::refusal(role, causes::FORM, path, &error))?;
        if value.prompts.is_empty() {
            return Err(crate::failures::refuse(
                &causes::NONE,
                &[
                    ("path", &path.display().to_string()),
                    ("state", &file::state(path)),
                ],
            ));
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
