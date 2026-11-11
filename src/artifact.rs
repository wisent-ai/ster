use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::failures::{pairs, refuse};
use crate::{chat, runtime::Precision};

pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContrastivePair {
    pub positive: String,
    pub negative: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairSet {
    #[serde(default)]
    pub trait_name: String,
    pub pairs: Vec<ContrastivePair>,
}

impl PairSet {
    pub fn load(path: &Path) -> Result<Self> {
        let shown = path.display().to_string();
        let bytes = fs::read(path).map_err(|error| pairs::read_refusal(&shown, &error))?;
        let value: Self = serde_json::from_slice(&bytes).map_err(|error| {
            refuse(
                &pairs::MALFORMED,
                &[("path", &shown), ("error", &error.to_string())],
            )
        })?;
        value.validate(&shown)?;
        Ok(value)
    }

    /// Checks the two content invariants every consumer of a pair set relies on.
    ///
    /// `label` is the identity quoted in the refusal sentence. Callers that read
    /// from disk pass the path; callers that validate an in-memory set built from
    /// an API request pass whatever names it in the operator's mental model. The
    /// two causes below carry the sentences the runbook publishes, so they stay
    /// byte-identical inside the refusal regardless of which caller triggers them.
    pub fn validate(&self, label: &str) -> Result<()> {
        if self.pairs.is_empty() {
            return Err(refuse(&pairs::EMPTY, &[("label", label)]));
        }
        for (index, pair) in self.pairs.iter().enumerate() {
            let side = if pair.positive.trim().is_empty() {
                "positive"
            } else if pair.negative.trim().is_empty() {
                "negative"
            } else {
                continue;
            };
            return Err(refuse(
                &pairs::BLANK,
                &[
                    ("label", label),
                    ("index", &index.to_string()),
                    ("side", side),
                ],
            ));
        }
        Ok(())
    }

    /// Writes the pair set as pretty JSON with a trailing newline.
    ///
    /// Steering artifacts are compact because nobody reads them, but pair files are
    /// hand-edited and diffed in review, so the extra bytes buy a readable file and
    /// a one-line-per-change diff. The trailing newline keeps POSIX tooling happy.
    pub fn save(&self, path: &Path) -> Result<()> {
        let shown = path.display().to_string();
        self.validate(&shown)?;
        // `Path::parent` yields an empty path for a bare file name; creating that
        // directory fails, so only create a parent that actually names one.
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                refuse(
                    &pairs::DIRECTORY,
                    &[
                        ("path", &shown),
                        ("directory", &parent.display().to_string()),
                        ("error", &error.to_string()),
                    ],
                )
            })?;
        }
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        fs::write(path, bytes).map_err(|error| {
            refuse(
                &pairs::UNWRITABLE,
                &[("path", &shown), ("error", &error.to_string())],
            )
        })
    }
}

/// What a JSON document on disk turns out to be, recognised by the fields one
/// product's document carries and the other's never does.
///
/// Both halves of Ster write JSON an operator names with a flag, and on
/// `generate` the two flags sit beside each other. Handing one to the other
/// used to escape as serde's own message — "missing field `rank` at line 1
/// column 207003" — which names a field of the type that failed to parse, a
/// byte offset into a file nobody will open, and nothing about the mistake
/// that was actually made. Recognising the foreign document first costs one
/// `Value` parse on a path that is about to parse the same bytes anyway, and
/// buys a sentence naming both ends of it.
///
/// The distinguishing fields are the honest ones: a steering artifact is a
/// trait and a set of per-layer vectors, an adapter sidecar is a rank and a
/// set of projections, and neither has ever carried the other's pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Document {
    Steering,
    AdapterSidecar,
    /// Anything else, including bytes that are not JSON at all. Recognition
    /// only exists to redirect a confusable document; everything it does not
    /// recognise goes on to the real loader and gets the real parse error.
    Unrecognised,
}

impl Document {
    pub fn recognise(bytes: &[u8]) -> Self {
        let Ok(serde_json::Value::Object(fields)) = serde_json::from_slice(bytes) else {
            return Self::Unrecognised;
        };
        if fields.contains_key("trait_name") && fields.contains_key("vectors") {
            Self::Steering
        } else if fields.contains_key("rank") && fields.contains_key("targets") {
            Self::AdapterSidecar
        } else {
            Self::Unrecognised
        }
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
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read steering artifact {}", path.display()))?;
        if Document::recognise(&bytes) == Document::AdapterSidecar {
            bail!(
                "{} is a LoRA adapter sidecar, not a steering artifact: it carries rank and targets where a steering artifact carries trait_name and vectors",
                path.display()
            );
        }
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
            bail!(
                "artifact has no steering vectors: hidden size {}, {} vector(s)",
                self.hidden_size,
                self.vectors.len()
            );
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
