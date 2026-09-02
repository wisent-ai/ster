//! Low-rank adapters: the only weights Ster ever trains, and the document that carries them.
//!
//! Ster's base model is mapped read-only from safetensors and never enters a
//! [`VarMap`], so the tensors registered here are by construction the complete
//! trainable set. An adapter is a pair `(a, b)` with `a: [rank, in]` and
//! `b: [out, rank]`; the update applied to a projection is
//! `scale * x @ a^T @ b^T`, which is the low-rank factorisation written in the
//! order that never materialises the dense `out x in` product.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarMap};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

/// `count` draws from a normal distribution with mean zero and the given
/// standard deviation, taken from `rng` so the sequence is the seed's.
///
/// Box-Muller rather than a distribution crate: two uniforms make one normal
/// pair with no dependency, and the adapter draw is the only place Ster needs
/// a Gaussian.
fn normal_draw(rng: &mut StdRng, count: usize, stdev: f64) -> Vec<f32> {
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        // `f64::ln` of zero is negative infinity, so the first uniform is
        // pulled into (0, 1] before the logarithm sees it.
        let first: f64 = 1.0 - rng.random::<f64>();
        let second: f64 = rng.random::<f64>();
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f64::consts::TAU * second;
        values.push((stdev * radius * angle.cos()) as f32);
        if values.len() < count {
            values.push((stdev * radius * angle.sin()) as f32);
        }
    }
    values
}

/// Which projections carry adapters.
///
/// `Ord` is derived because [`Adapters`] keys a [`BTreeMap`] by
/// `(layer, target)`, which is what makes adapter construction — and therefore
/// the sequence of draws from the random initialiser — reproducible for a seed.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Query,
    Key,
    Value,
    Output,
    Gate,
    Up,
    Down,
}

impl Target {
    /// Every target, in the order the refusal sentences and the CLI list them.
    pub const ALL: [Target; 7] = [
        Target::Query,
        Target::Key,
        Target::Value,
        Target::Output,
        Target::Gate,
        Target::Up,
        Target::Down,
    ];

    /// Reads a target from operator input.
    ///
    /// Both spellings are accepted: the short name Ster prints, and the
    /// Hugging Face projection name operators read off a model card. Neither is
    /// ambiguous, and refusing `q_proj` would only teach a translation table.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "query" | "q_proj" | "q" => Ok(Self::Query),
            "key" | "k_proj" | "k" => Ok(Self::Key),
            "value" | "v_proj" | "v" => Ok(Self::Value),
            "output" | "o_proj" | "o" => Ok(Self::Output),
            "gate" | "gate_proj" => Ok(Self::Gate),
            "up" | "up_proj" => Ok(Self::Up),
            "down" | "down_proj" => Ok(Self::Down),
            _ => bail!(
                "unknown adapter target {text:?}; expected one of query, key, value, output, gate, up, down"
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::Output => "output",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    /// The projection's weight name inside a Hugging Face Llama checkpoint.
    ///
    /// This is the other half of the naming contract: [`Adapter::tensor_names`]
    /// says where a factor lives in a Ster artifact, and this says which base
    /// tensor that factor is an update to. Merging is the one operation that
    /// needs both, and hard-coding the mapping at its call site would put half
    /// the contract in a file that does not own it.
    pub fn checkpoint_tensor(self, layer: usize) -> String {
        let leaf = match self {
            Self::Query => "self_attn.q_proj",
            Self::Key => "self_attn.k_proj",
            Self::Value => "self_attn.v_proj",
            Self::Output => "self_attn.o_proj",
            Self::Gate => "mlp.gate_proj",
            Self::Up => "mlp.up_proj",
            Self::Down => "mlp.down_proj",
        };
        format!("model.layers.{layer}.{leaf}.weight")
    }

    /// The `(outputs, inputs)` shape of the projection this target adapts.
    ///
    /// Grouped-query attention makes key and value narrower than query, and the
    /// feed-forward block is wider than the residual stream, so the adapter
    /// factors are not square and cannot be derived from `hidden` alone.
    fn widths(self, hidden: usize, kv_width: usize, intermediate: usize) -> (usize, usize) {
        match self {
            Self::Query | Self::Output => (hidden, hidden),
            Self::Key | Self::Value => (kv_width, hidden),
            Self::Gate | Self::Up => (intermediate, hidden),
            Self::Down => (hidden, intermediate),
        }
    }

    /// Whether the projection reads the residual stream, so `a` is `[rank, hidden_size]`.
    fn reads_hidden(self) -> bool {
        !matches!(self, Self::Down)
    }

    /// Whether the projection writes the residual stream, so `b` is `[hidden_size, rank]`.
    fn writes_hidden(self) -> bool {
        matches!(self, Self::Query | Self::Output | Self::Down)
    }
}

#[derive(Debug, Clone)]
pub struct Spec {
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<Target>,
    pub layers: Vec<usize>,
    /// Seeds the draw that fills `A`. Two runs of the same command must produce
    /// the same adapter, and Candle's initialiser draws from the device's own
    /// generator, so the seed has to reach the device before the first tensor
    /// is created rather than only shuffling the example order later.
    pub seed: u64,
}

impl Spec {
    /// Checks everything a spec can get wrong before a model is touched.
    ///
    /// An empty `layers` list is deliberately valid and means "every layer":
    /// the CLI parses `--layers all` before it knows how many layers the model
    /// has, so it hands the emptiness down and [`Spec::resolved`] expands it
    /// once the count is known. An empty `targets` list has no such excuse.
    pub fn validate(&self, layer_count: usize) -> Result<()> {
        if self.rank == 0 {
            bail!("adapter rank 0 trains nothing; choose a rank of at least 1");
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            bail!("adapter alpha {} is not a positive finite number", self.alpha);
        }
        if self.targets.is_empty() {
            bail!(
                "adapter spec names no targets; choose at least one of query, key, value, output, gate, up, down"
            );
        }
        for (index, target) in self.targets.iter().enumerate() {
            if self.targets[..index].contains(target) {
                bail!("adapter spec names target {} twice", target.name());
            }
        }
        for layer in &self.layers {
            if *layer >= layer_count {
                bail!(
                    "adapter spec names layer {layer}, but the model has {layer_count} layers"
                );
            }
        }
        Ok(())
    }

    /// The spec with `layers` pinned to concrete indices.
    ///
    /// Callers that hold the layer count run this once and pass the result
    /// everywhere, so the adapters that get built and the layer list recorded in
    /// the artifact can never disagree. Sorting and de-duplicating also fixes the
    /// order in which random factors are drawn, which is what makes `--seed`
    /// mean something.
    pub fn resolved(&self, layer_count: usize) -> Self {
        let mut layers = if self.layers.is_empty() {
            (0..layer_count).collect::<Vec<_>>()
        } else {
            self.layers.clone()
        };
        layers.sort_unstable();
        layers.dedup();
        Self {
            rank: self.rank,
            alpha: self.alpha,
            targets: self.targets.clone(),
            layers,
            seed: self.seed,
        }
    }

    /// The constant the low-rank product is multiplied by, `alpha / rank`.
    ///
    /// Dividing by the rank is what lets an operator raise the rank without also
    /// raising the effective learning rate of the update.
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}

/// One low-rank pair. `b` starts at zero so an untrained adapter is the identity.
#[derive(Debug, Clone)]
pub struct Adapter {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f64,
}

impl Adapter {
    /// The on-disk names of the two factors. This string is the artifact's contract.
    pub fn tensor_names(layer: usize, target: Target) -> (String, String) {
        let target = target.name();
        (format!("layers.{layer}.{target}.a"), format!("layers.{layer}.{target}.b"))
    }

    /// The low-rank update for `xs`, shaped `[batch, sequence, in]`.
    ///
    /// The two matmuls run in the order `x -> rank -> out`, so the widest thing
    /// ever allocated is `[batch * sequence, rank]`; folding `a` into `b` first
    /// would build the dense `out x in` matrix this whole scheme exists to avoid.
    /// The factors are cast to `xs`'s dtype and device rather than the reverse:
    /// an artifact trained in F32 may be attached to a model running in another
    /// dtype, and moving `rank * width` values is orders of magnitude cheaper
    /// than moving the activation. Both casts are a plain `clone` when they
    /// already agree, which is the training case, and a clone keeps the tensor
    /// id, so the gradient still reaches the registered variable.
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, sequence, width) = xs.dims3()?;
        let a = self.a.to_device(xs.device())?.to_dtype(xs.dtype())?;
        let b = self.b.to_device(xs.device())?.to_dtype(xs.dtype())?;
        let outputs = b.dim(0)?;
        // Flattening the batch away turns the update into two plain 2-D matmuls.
        // The width comes from `xs`, not from `a`, so a factor that does not fit
        // this projection fails in the matmul with a shape error that names both
        // operands instead of silently reinterpreting the activation.
        let flat = xs.reshape((batch * sequence, width))?;
        let low = flat.matmul(&a.t()?)?;
        let full = low.matmul(&b.t()?)?;
        (full * self.scale)?.reshape((batch, sequence, outputs))
    }
}

/// Adapters for the whole model, keyed by (layer, target).
#[derive(Debug, Clone, Default)]
pub struct Adapters {
    entries: BTreeMap<(usize, Target), Adapter>,
}

impl Adapters {
    pub fn get(&self, layer: usize, target: Target) -> Option<&Adapter> {
        self.entries.get(&(layer, target))
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many `(layer, target)` sites carry an adapter.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The factors under their on-disk names, ready to become an [`Artifact`].
    ///
    /// Building the map here rather than at the call site keeps the naming
    /// contract in one place: nothing outside this module has to know that a
    /// tensor is called `layers.{layer}.{target}.a`.
    pub fn tensors(&self) -> BTreeMap<String, Tensor> {
        let mut tensors = BTreeMap::new();
        for ((layer, target), adapter) in &self.entries {
            let (a_name, b_name) = Adapter::tensor_names(*layer, *target);
            tensors.insert(a_name, adapter.a.clone());
            tensors.insert(b_name, adapter.b.clone());
        }
        tensors
    }

    /// Fresh trainable adapters registered in `varmap`; A is normal(0, 1/rank), B is zeros.
    ///
    /// Zeroing `b` means the update starts at exactly zero, so the first training
    /// step sees the base model's own behaviour rather than noise injected into
    /// every projection. Registering through [`VarMap::get`] is what makes
    /// `VarMap::all_vars` return the trainable set and nothing else: the base
    /// weights arrive from a mmap'd `VarBuilder` and are never registered.
    pub fn fresh(
        spec: &Spec,
        varmap: &VarMap,
        hidden: usize,
        kv_width: usize,
        intermediate: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if spec.layers.is_empty() {
            bail!("adapter spec names no layers; resolve it against the model before building adapters");
        }
        if spec.targets.is_empty() {
            bail!(
                "adapter spec names no targets; choose at least one of query, key, value, output, gate, up, down"
            );
        }
        // Candle's initialisers draw from the device's generator, and the CPU
        // device refuses `set_seed` outright, so a run could not reproduce its
        // own adapter. The draw is done here instead, from an RNG this crate
        // seeds, and written into the registered variable — same tensors, same
        // `VarMap`, but the same command now yields the same adapter on every
        // device.
        let mut rng = StdRng::seed_from_u64(spec.seed);
        let scale = spec.scale();
        let mut layers = spec.layers.clone();
        layers.sort_unstable();
        layers.dedup();
        let mut entries = BTreeMap::new();
        for layer in layers {
            for target in &spec.targets {
                let (outputs, inputs) = target.widths(hidden, kv_width, intermediate);
                let (a_name, b_name) = Adapter::tensor_names(layer, *target);
                let a = varmap
                    .get((spec.rank, inputs), &a_name, Init::Const(0.0), dtype, device)
                    .with_context(|| format!("failed to create adapter tensor {a_name}"))?;
                let draw = normal_draw(&mut rng, spec.rank * inputs, 1.0 / spec.rank as f64);
                let seeded = Tensor::from_vec(draw, (spec.rank, inputs), device)
                    .and_then(|tensor| tensor.to_dtype(dtype))
                    .with_context(|| format!("failed to draw adapter tensor {a_name}"))?;
                varmap
                    .data()
                    .lock()
                    .expect("adapter variable map lock")
                    .get(&a_name)
                    .with_context(|| format!("adapter tensor {a_name} was not registered"))?
                    .set(&seeded)
                    .with_context(|| format!("failed to initialise adapter tensor {a_name}"))?;
                let b = varmap
                    .get((outputs, spec.rank), &b_name, Init::Const(0.0), dtype, device)
                    .with_context(|| format!("failed to create adapter tensor {b_name}"))?;
                entries.insert((layer, *target), Adapter { a, b, scale });
            }
        }
        Ok(Self { entries })
    }

    /// Frozen adapters read from an artifact, for inference.
    ///
    /// Nothing here touches a [`VarMap`]: these tensors are constants attached to
    /// a forward pass, and generation must not be able to change them.
    pub fn from_artifact(artifact: &Artifact, device: &Device, dtype: DType) -> Result<Self> {
        artifact.validate()?;
        let scale = artifact.alpha / artifact.rank as f64;
        let mut entries = BTreeMap::new();
        for layer in &artifact.layers {
            for target in &artifact.targets {
                let (a_name, b_name) = Adapter::tensor_names(*layer, *target);
                let a = artifact
                    .tensors
                    .get(&a_name)
                    .with_context(|| format!("adapter artifact is missing tensor {a_name}"))?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                let b = artifact
                    .tensors
                    .get(&b_name)
                    .with_context(|| format!("adapter artifact is missing tensor {b_name}"))?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                entries.insert((*layer, *target), Adapter { a, b, scale });
            }
        }
        Ok(Self { entries })
    }
}

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
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
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
            bail!("adapter artifact belongs to product {:?}, not Ster", self.product);
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
        bail!("adapter tensor {name} has {} dimensions, expected 2", dims.len());
    }
    if dims[axis] != expected {
        let axis_name = if axis == 0 { "rows" } else { "columns" };
        bail!("adapter tensor {name} has {} {axis_name}, expected {expected}", dims[axis]);
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
