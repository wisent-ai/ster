//! The documents Ster reads and writes beside a model: a contrastive pair
//! set, and the steering artifact a trained direction travels in.

use std::{fs, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::failures::{file, pairs, refuse};

/// What a pair set is called in a refusal of its file.
const PAIR_SET: &str = "the pair set";

mod steering;

pub use steering::{LayerVector, SteeringArtifact};

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
        let bytes = fs::read(path)
            .map_err(|error| file::read::refusal_or(&pairs::MISSING, PAIR_SET, path, &error))?;
        let value: Self = serde_json::from_slice(&bytes)
            .map_err(|error| file::json::refusal(PAIR_SET, pairs::FORM, path, &error))?;
        value.validate(&path.display().to_string())?;
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
            return Err(refuse(&pairs::EMPTY, &[("label", label), ("trait", &self.trait_name)]));
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
        self.validate(&path.display().to_string())?;
        // `Path::parent` yields an empty path for a bare file name; creating that
        // directory fails, so only create a parent that actually names one.
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                let left = "nothing was written";
                file::write::refusal("creating the directory of", PAIR_SET, path, left, &error)
            })?;
        }
        let mut bytes = crate::json::pretty(self).into_bytes();
        bytes.push(b'\n');
        let mut opened = fs::File::create(path).map_err(|error| {
            let left = "nothing was written, and a set already there is unchanged";
            file::write::refusal("opening for writing", PAIR_SET, path, left, &error)
        })?;
        std::io::Write::write_all(&mut opened, &bytes).map_err(|error| {
            let left = "the file was emptied and holds only part of the set";
            file::write::refusal("writing", PAIR_SET, path, left, &error)
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
