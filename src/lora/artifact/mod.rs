//! The document a trained adapter travels in: what it declares, the tensors
//! beside it, and every check a reader runs before trusting them.

mod check;

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use super::Target;
use crate::failures::{
    file,
    lora::artifact::{ADAPTER_MISSING, SIDECAR_FORM, SIDECAR_MISSING, STEERING_GIVEN},
    refuse,
};

/// What an adapter's safetensors file is called in a refusal of it.
const WEIGHTS: &str = "the adapter's weights";

/// What an adapter's JSON sidecar is called in a refusal of it.
const SIDECAR: &str = "the adapter's sidecar";

pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// The name of the scalar reward head inside a reward artifact's safetensors.
/// This string is part of the artifact's contract, exactly as the adapter
/// factor names are.
pub const REWARD_HEAD_TENSOR: &str = "reward.head";

/// What an artifact is for.
///
/// A reward model is an adapter *and* a scalar head trained on top of it, and
/// the two are useless apart: the head reads a residual stream the adapters
/// shaped, and the adapters were shaped to make that head separate. They
/// therefore travel in one file, and the kind is what tells a reader which of
/// the two things it is holding — so `generate` can refuse a reward model
/// rather than silently apply half of one.
///
/// The field defaults to [`Kind::Adapter`] so that every sidecar written
/// before it existed still loads and still means what it meant. It is
/// additive, which is why the schema version does not move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    #[default]
    Adapter,
    Reward,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Adapter => "adapter",
            Self::Reward => "reward",
        }
    }
}

/// The durable adapter document.
///
/// The weights go in a safetensors file so any other tool can read them, and the
/// identity — which model, which revision, which rank — goes in a JSON sidecar
/// beside it, exactly as [`crate::artifact::SteeringArtifact`] carries identity.
/// Splitting them is what lets Ster refuse an adapter trained against a
/// different model before a single tensor is loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub schema_version: u32,
    pub product: String,
    #[serde(default)]
    pub kind: Kind,
    pub model: String,
    pub model_revision: Option<String>,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<Target>,
    pub layers: Vec<usize>,
    pub hidden_size: usize,
    pub train: serde_json::Value,
    #[serde(skip)]
    pub tensors: BTreeMap<String, Tensor>,
}

impl Artifact {
    /// The sidecar that sits beside the safetensors file.
    ///
    /// The extension is replaced rather than appended, so `x.lora.safetensors`
    /// becomes `x.lora.json` and the pair sorts together in a directory listing.
    pub fn sidecar_path(path: &Path) -> PathBuf {
        path.with_extension("json")
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        // `Path::parent` yields an empty path for a bare file name; creating that
        // directory fails, so only create a parent that actually names one.
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                let left = "no adapter was written";
                file::write::refusal("creating the directory of", WEIGHTS, path, left, &error)
            })?;
        }
        let tensors: HashMap<String, Tensor> = self
            .tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        candle_core::safetensors::save(&tensors, path).map_err(|error| {
            let left = "the weights file may hold part of the adapter, and no sidecar was written";
            file::tensors::write_refusal(WEIGHTS, path, left, &error)
        })?;
        let sidecar = Self::sidecar_path(path);
        let mut bytes = crate::json::pretty(self).into_bytes();
        bytes.push(b'\n');
        fs::write(&sidecar, bytes).map_err(|error| {
            let left = "the weights file was written and its sidecar was not, so the adapter does not load until it is saved again";
            file::write::refusal("writing", SIDECAR, &sidecar, left, &error)
        })
    }

    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let shown = path.display().to_string();
        let sidecar = Self::sidecar_path(path);
        let sidecar_shown = sidecar.display().to_string();
        // Preserve filesystem inspection errors instead of reporting an
        // unreadable artifact as absent.
        match path.try_exists() {
            Ok(true) => {}
            Ok(false) => {
                return Err(refuse(
                    &ADAPTER_MISSING,
                    &[
                        ("path", &shown),
                        ("state", &file::state(path)),
                        ("parent", &file::parent(path).display().to_string()),
                    ],
                ));
            }
            Err(error) => return Err(file::read::refusal(WEIGHTS, path, &error)),
        }
        let bytes = match fs::read(&sidecar) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Err(refuse(
                    &SIDECAR_MISSING,
                    &[("path", &shown), ("sidecar", &sidecar_shown)],
                ));
            }
            Err(error) => return Err(file::read::refusal(SIDECAR, &sidecar, &error)),
        };
        // The mirror of the refusal in `SteeringArtifact::load`, and written
        // as one: `generate` takes `--vector` and `--adapter` side by side,
        // and an operator who crosses them deserves the same sentence from
        // whichever end they crossed.
        if crate::artifact::Document::recognise(&bytes) == crate::artifact::Document::Steering {
            return Err(refuse(
                &STEERING_GIVEN,
                &[("path", &shown), ("sidecar", &sidecar_shown)],
            ));
        }
        let mut value: Self = serde_json::from_slice(&bytes)
            .map_err(|error| file::json::refusal(SIDECAR, SIDECAR_FORM, &sidecar, &error))?;
        let tensors = candle_core::safetensors::load(path, device)
            .map_err(|error| file::tensors::read_refusal(WEIGHTS, path, &error))?;
        value.tensors = tensors.into_iter().collect();
        value.validate().map_err(|error| {
            crate::failures::amend(error, format!("sidecar {sidecar_shown}, weights {shown}"))
        })?;
        Ok(value)
    }
}
