//! The live weights: one low-rank pair per projection, the draw that fills
//! them, and the set a forward pass looks them up in.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarMap};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{Spec, Target, Widths, artifact::Artifact};

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
        (
            format!("layers.{layer}.{target}.a"),
            format!("layers.{layer}.{target}.b"),
        )
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
        widths: Widths,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if spec.layers.is_empty() {
            bail!(
                "adapter spec names no layers; resolve it against the model before building adapters"
            );
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
                let (outputs, inputs) = target.widths(widths);
                let (a_name, b_name) = Adapter::tensor_names(layer, *target);
                let a = varmap
                    .get(
                        (spec.rank, inputs),
                        &a_name,
                        Init::Const(0.0),
                        dtype,
                        device,
                    )
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
                    .get(
                        (outputs, spec.rank),
                        &b_name,
                        Init::Const(0.0),
                        dtype,
                        device,
                    )
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
