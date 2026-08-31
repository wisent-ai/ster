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
        value.validate(&path.display().to_string())?;
        Ok(value)
    }

    /// Checks the two content invariants every consumer of a pair set relies on.
    ///
    /// `label` is the identity quoted in the refusal sentence. Callers that read
    /// from disk pass the path; callers that validate an in-memory set built from
    /// an API request pass whatever names it in the operator's mental model. The
    /// two sentences below are published in the runbook, so they must stay
    /// byte-identical regardless of which caller triggers them.
    pub fn validate(&self, label: &str) -> Result<()> {
        if self.pairs.is_empty() {
            bail!("pair set {label} contains no pairs");
        }
        if self
            .pairs
            .iter()
            .any(|pair| pair.positive.trim().is_empty() || pair.negative.trim().is_empty())
        {
            bail!("pair set {label} contains an empty positive or negative prompt");
        }
        Ok(())
    }

    /// Writes the pair set as pretty JSON with a trailing newline.
    ///
    /// Steering artifacts are compact because nobody reads them, but pair files are
    /// hand-edited and diffed in review, so the extra bytes buy a readable file and
    /// a one-line-per-change diff. The trailing newline keeps POSIX tooling happy.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate(&path.display().to_string())?;
        // `Path::parent` yields an empty path for a bare file name; creating that
        // directory fails, so only create a parent that actually names one.
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        fs::write(path, bytes)
            .with_context(|| format!("failed to write pair set {}", path.display()))
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
