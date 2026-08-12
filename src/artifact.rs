use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastivePair {
    pub positive: String,
    pub negative: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairSet {
    #[serde(default)]
    pub trait_name: String,
    pub pairs: Vec<ContrastivePair>,
}

impl PairSet {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read pair set {}", path.display()))?;
        let value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid pair set JSON in {}", path.display()))?;
        if value.pairs.is_empty() {
            bail!("pair set {} contains no pairs", path.display());
        }
        if value
            .pairs
            .iter()
            .any(|pair| pair.positive.trim().is_empty() || pair.negative.trim().is_empty())
        {
            bail!("pair set {} contains an empty positive or negative prompt", path.display());
        }
        Ok(value)
    }
}

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
    ) -> Self {
        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            product: "ster".to_owned(),
            model,
            model_revision,
            trait_name,
            method,
            hidden_size,
            vectors,
            metadata: BTreeMap::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read steering artifact {}", path.display()))?;
        let value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid steering artifact JSON in {}", path.display()))?;
        value.validate()?;
        Ok(value)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let bytes = serde_json::to_vec(self)?;
        fs::write(path, bytes)
            .with_context(|| format!("failed to write steering artifact {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ARTIFACT_SCHEMA_VERSION {
            bail!(
                "artifact schema {} is unsupported; this Ster build reads schema {}",
                self.schema_version,
                ARTIFACT_SCHEMA_VERSION
            );
        }
        if self.product != "ster" {
            bail!("artifact belongs to product {:?}, not Ster", self.product);
        }
        if self.hidden_size == 0 || self.vectors.is_empty() {
            bail!("artifact has no steering vectors");
        }
        for vector in &self.vectors {
            if vector.values.len() != self.hidden_size {
                bail!(
                    "layer {} has vector width {}, expected {}",
                    vector.layer,
                    vector.values.len(),
                    self.hidden_size
                );
            }
            if vector.values.iter().any(|value| !value.is_finite()) {
                bail!("layer {} contains a non-finite value", vector.layer);
            }
        }
        Ok(())
    }
}
