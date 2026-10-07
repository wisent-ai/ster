//! The document a trained adapter travels in: what it declares, the tensors
//! beside it, and every check a reader runs before trusting them.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use super::{Target, adapter::Adapter};

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
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let tensors: HashMap<String, Tensor> = self
            .tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        candle_core::safetensors::save(&tensors, path)
            .with_context(|| format!("failed to write adapter weights {}", path.display()))?;
        let sidecar = Self::sidecar_path(path);
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        fs::write(&sidecar, bytes)
            .with_context(|| format!("failed to write adapter sidecar {}", sidecar.display()))
    }

    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        // The weights are named by the operator, the sidecar is derived from
        // them, so an absent artifact must say so about the path that was
        // actually asked for. Checking the sidecar first reported a missing
        // sidecar for a path that had no weights either, which sends the
        // reader looking for the wrong file.
        if !path.exists() {
            bail!("failed to read adapter {}", path.display());
        }
        let sidecar = Self::sidecar_path(path);
        if !sidecar.exists() {
            bail!(
                "adapter {} has no sidecar at {}; the pair is written together and must travel together",
                path.display(),
                sidecar.display()
            );
        }
        let bytes = fs::read(&sidecar)
            .with_context(|| format!("failed to read adapter sidecar {}", sidecar.display()))?;
        // The mirror of the refusal in `SteeringArtifact::load`, and written
        // as one: `generate` takes `--vector` and `--adapter` side by side,
        // and an operator who crosses them deserves the same sentence from
        // whichever end they crossed.
        if crate::artifact::Document::recognise(&bytes) == crate::artifact::Document::Steering {
            bail!(
                "{} is a steering artifact, not a LoRA adapter sidecar: it carries trait_name and vectors where an adapter sidecar carries rank and targets",
                sidecar.display()
            );
        }
        let mut value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid adapter sidecar JSON in {}", sidecar.display()))?;
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("failed to read adapter weights {}", path.display()))?;
        value.tensors = tensors.into_iter().collect();
        value.validate()?;
        Ok(value)
    }

    /// Refuses every artifact a forward pass could not honour.
    ///
    /// Shape checks are as tight as the document allows: the sidecar records
    /// `hidden_size` but not the grouped-query or feed-forward widths, so a
    /// key or gate factor can be checked against `rank` alone while a query or
    /// down factor is checked against both.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ARTIFACT_SCHEMA_VERSION {
            bail!(
                "adapter artifact schema {} is unsupported; this Ster build reads schema {}",
                self.schema_version,
                ARTIFACT_SCHEMA_VERSION
            );
        }
        if self.product != "ster" {
            bail!(
                "adapter artifact belongs to product {:?}, not Ster",
                self.product
            );
        }
        if self.rank == 0 {
            bail!("adapter artifact declares rank 0, which stores nothing");
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            bail!(
                "adapter artifact declares alpha {}, which is not a positive finite number",
                self.alpha
            );
        }
        if self.hidden_size == 0 {
            bail!("adapter artifact declares hidden size 0");
        }
        if self.targets.is_empty() {
            bail!("adapter artifact names no targets");
        }
        if self.layers.is_empty() {
            bail!("adapter artifact names no layers");
        }
        for (index, target) in self.targets.iter().enumerate() {
            if self.targets[..index].contains(target) {
                bail!("adapter artifact names target {} twice", target.name());
            }
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if self.layers[..index].contains(layer) {
                bail!("adapter artifact names layer {layer} twice");
            }
        }
        for layer in &self.layers {
            for target in &self.targets {
                let (a_name, b_name) = Adapter::tensor_names(*layer, *target);
                let a = self
                    .tensors
                    .get(&a_name)
                    .with_context(|| format!("adapter artifact is missing tensor {a_name}"))?;
                let b = self
                    .tensors
                    .get(&b_name)
                    .with_context(|| format!("adapter artifact is missing tensor {b_name}"))?;
                check_factor(&a_name, a, 0, self.rank)?;
                if target.reads_hidden() {
                    check_factor(&a_name, a, 1, self.hidden_size)?;
                }
                check_factor(&b_name, b, 1, self.rank)?;
                if target.writes_hidden() {
                    check_factor(&b_name, b, 0, self.hidden_size)?;
                }
                for (name, tensor) in [(&a_name, a), (&b_name, b)] {
                    finite(name, tensor)?;
                }
            }
        }
        // The head is the whole point of a reward artifact and meaningless in
        // a plain one, so both directions are refused: a reward document
        // without a head would load as a generation adapter that scores
        // nothing, and an adapter carrying one would have come from somewhere
        // this build cannot account for.
        match (self.kind, self.tensors.get(REWARD_HEAD_TENSOR)) {
            (Kind::Reward, Some(head)) => {
                check_factor(REWARD_HEAD_TENSOR, head, 0, 1)?;
                check_factor(REWARD_HEAD_TENSOR, head, 1, self.hidden_size)?;
                finite(REWARD_HEAD_TENSOR, head)?;
            }
            (Kind::Reward, None) => bail!(
                "reward artifact is missing tensor {REWARD_HEAD_TENSOR}, which is the head it exists to carry"
            ),
            (Kind::Adapter, Some(_)) => bail!(
                "adapter artifact carries a {REWARD_HEAD_TENSOR} tensor but declares kind adapter"
            ),
            (Kind::Adapter, None) => {}
        }
        Ok(())
    }
}

/// Checks one dimension of one factor, naming the axis the way a reader thinks of it.
fn check_factor(name: &str, tensor: &Tensor, axis: usize, expected: usize) -> Result<()> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        bail!(
            "adapter tensor {name} has {} dimensions, expected 2",
            dims.len()
        );
    }
    if dims[axis] != expected {
        let axis_name = if axis == 0 { "rows" } else { "columns" };
        bail!(
            "adapter tensor {name} has {} {axis_name}, expected {expected}",
            dims[axis]
        );
    }
    Ok(())
}

/// Refuses a tensor with a value no forward pass could survive.
///
/// A non-finite factor does not fail loudly at matmul time: it propagates a
/// `NaN` through the residual stream and comes out as a plausible-looking
/// decode, so it is caught when the document is read rather than when the
/// damage is visible.
fn finite(name: &str, tensor: &Tensor) -> Result<()> {
    let values = tensor
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .with_context(|| format!("failed to read adapter tensor {name}"))?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("adapter tensor {name} contains a non-finite value");
    }
    Ok(())
}
