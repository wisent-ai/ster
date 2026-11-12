//! The steering artifact: the JSON document a trained direction travels in,
//! one vector per layer it steers, and the provenance of the run that fitted
//! it.

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{ARTIFACT_SCHEMA_VERSION, Document};
use crate::failures::{file, refuse, steering};
use crate::{chat, runtime::Precision};

/// What a steering artifact is called in a refusal of its file.
const ROLE: &str = "the steering artifact";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerVector {
    pub layer: usize,
    pub values: Vec<f32>,
    pub train_margin: f32,
    pub train_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringArtifact {
    pub schema_version: u32,
    pub product: String,
    pub model: String,
    pub model_revision: Option<String>,
    /// The dtype the base weights were mapped at when this direction was
    /// fitted, spelled exactly as `--precision` spells it: `"f32"`, `"f16"`,
    /// or `"bf16"`.
    ///
    /// `evaluate --precision` has always told the operator it should match
    /// the run that trained the artifact, and until this field existed
    /// nothing could check it — the claim was advice with no evidence behind
    /// it. An adapter sidecar has recorded the same value at
    /// `train.precision` since precision landed; this is the steering half of
    /// the same provenance, in the same spelling, so one comparison reads
    /// both.
    ///
    /// `None` means an artifact written before this field existed. It is
    /// additive and defaulted for exactly that reason, which is also why the
    /// schema version does not move. It is deliberately not a `metadata`
    /// entry: provenance the product wrote must stay distinguishable from
    /// notes the operator wrote.
    #[serde(default)]
    pub precision: Option<String>,
    /// Whether the pairs this direction was fitted from were read through the
    /// model's own chat template: `"applied"`, `"absent"`, or `"off"`, the
    /// same three words an adapter sidecar records at `train.chat_template`.
    ///
    /// The same provenance argument as `precision`, and the sharper half of
    /// it. Precision changes where a direction points by a few ulps; format
    /// changes what the residual stream contains, because a templated prompt
    /// is a user turn the assistant is about to answer and a raw one is a
    /// document being continued. A direction fitted `off` and added during an
    /// `applied` decode is measured in one space and steers another, and
    /// nothing about the artifact used to say so.
    ///
    /// `None` means an artifact written before this field existed.
    #[serde(default)]
    pub chat_template: Option<String>,
    pub trait_name: String,
    pub method: String,
    pub hidden_size: usize,
    pub vectors: Vec<LayerVector>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl SteeringArtifact {
    pub fn new(
        model: String,
        model_revision: Option<String>,
        trait_name: String,
        method: String,
        hidden_size: usize,
        vectors: Vec<LayerVector>,
        precision: Precision,
        chat_template: chat::Status,
    ) -> Self {
        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            product: "ster".to_owned(),
            model,
            model_revision,
            precision: Some(precision.name().to_owned()),
            chat_template: Some(chat_template.label().to_owned()),
            trait_name,
            method,
            hidden_size,
            vectors,
            metadata: BTreeMap::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|error| {
            file::read::refusal_or(&steering::MISSING, ROLE, path, &error)
        })?;
        if Document::recognise(&bytes) == Document::AdapterSidecar {
            return Err(refuse(
                &steering::ADAPTER_GIVEN,
                &[("path", &path.display().to_string())],
            ));
        }
        let value: Self = serde_json::from_slice(&bytes)
            .map_err(|error| file::json::refusal(ROLE, steering::FORM, path, &error))?;
        value.validate()?;
        Ok(value)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let left = "no artifact was written there";
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|error| {
                file::write::refusal("creating the directory of", ROLE, path, left, &error)
            })?;
        }
        let bytes = crate::json::compact(self).into_bytes();
        fs::write(path, bytes)
            .map_err(|error| file::write::refusal("writing", ROLE, path, left, &error))
    }

    pub fn validate(&self) -> Result<()> {
        let trait_name = self.trait_name.as_str();
        if self.schema_version != ARTIFACT_SCHEMA_VERSION {
            return Err(refuse(
                &steering::SCHEMA_UNSUPPORTED,
                &[
                    ("trait_name", trait_name),
                    ("schema", &self.schema_version.to_string()),
                    ("supported", &ARTIFACT_SCHEMA_VERSION.to_string()),
                ],
            ));
        }
        if self.product != "ster" {
            return Err(refuse(
                &steering::PRODUCT_FOREIGN,
                &[("trait_name", trait_name), ("product", &format!("{:?}", self.product))],
            ));
        }
        if self.hidden_size == 0 || self.vectors.is_empty() {
            return Err(refuse(
                &steering::VECTORS_NONE,
                &[
                    ("trait_name", trait_name),
                    ("hidden_size", &self.hidden_size.to_string()),
                    ("count", &self.vectors.len().to_string()),
                ],
            ));
        }
        for vector in &self.vectors {
            let layer = vector.layer.to_string();
            if vector.values.len() != self.hidden_size {
                return Err(refuse(
                    &steering::VECTOR_WIDTH,
                    &[
                        ("trait_name", trait_name),
                        ("layer", &layer),
                        ("width", &vector.values.len().to_string()),
                        ("hidden_size", &self.hidden_size.to_string()),
                    ],
                ));
            }
            if vector.values.iter().any(|value| !value.is_finite()) {
                return Err(refuse(
                    &steering::VECTOR_NOT_FINITE,
                    &[("trait_name", trait_name), ("layer", &layer)],
                ));
            }
        }
        Ok(())
    }
}
