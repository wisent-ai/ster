//! Bringing a checkpoint into memory: the loaders an operator reaches, what
//! the three of them share, and the artifacts that may be attached on the way.

use std::path::Path;

use anyhow::{Result, bail};
use candle_nn::VarMap;

use crate::{lora, model::SteeringLlama};

use super::{DeviceChoice, Runtime, device::Precision, validate_layers};

mod base;
mod checkpoint;
mod config;
mod family;
mod gguf;
mod hy_v4;
mod mistral;

use base::BaseLoad;
pub use checkpoint::Checkpoint;
pub use mistral::Layout;

impl Runtime {
    /// Loads the frozen base model with no adapters attached, in F32.
    pub fn load(model: &str, revision: Option<&str>, device: DeviceChoice) -> Result<Self> {
        Self::load_at(model, revision, device, Precision::F32)
    }

    /// The same load at a stated precision.
    ///
    /// A separate entry point rather than a fifth argument on `load`, because
    /// the dtype is fixed while the weights are mapped and cannot be a setter
    /// the way the chat template is — and because every caller that has no
    /// opinion should keep mapping F32 without being asked to say so. The four
    /// loaders below pair the same way.
    pub fn load_at(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        precision: Precision,
    ) -> Result<Self> {
        let base = BaseLoad::resolve(model, revision, device, precision)?;
        let builder = base.builder()?;
        let model_impl =
            SteeringLlama::load(builder, base.config.clone(), base.architecture.clone())?;
        Ok(base.finish(model, model_impl))
    }

    /// Loads a model with frozen LoRA adapters read from an artifact.
    ///
    /// The identity checks mirror the ones `generate` runs against a steering
    /// artifact: an adapter trained against a different checkpoint or a
    /// different width is not merely inaccurate, it is a shape error waiting
    /// to surface halfway through a decode, so it is refused at load time.
    pub fn load_with_adapter(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        adapter: &Path,
    ) -> Result<Self> {
        Self::load_with_adapter_at(model, revision, device, adapter, Precision::F32)
    }

    /// The same load at a stated precision. A frozen adapter is cast to the
    /// base dtype on the way in, so an artifact trained in F32 attaches to a
    /// half-precision model and the whole graph stays one dtype — there is
    /// nothing to keep in F32 here, because nothing is being stepped.
    pub fn load_with_adapter_at(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        adapter: &Path,
        precision: Precision,
    ) -> Result<Self> {
        Ok(Self::load_artifact_at(
            model,
            revision,
            device,
            adapter,
            lora::Kind::Adapter,
            precision,
        )?
        .0)
    }

    /// The same load, for an artifact that must be of a stated `kind`, giving
    /// the caller the artifact back as well.
    ///
    /// A reward model's scalar head lives in the same file as its adapters, so
    /// the caller that wants the head needs the document the adapters came out
    /// of — handing it back beats reading the file twice. The kind is required
    /// rather than reported: an artifact that says what it is only helps if
    /// applying it somewhere it does not belong is a refusal.
    pub fn load_artifact(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        path: &Path,
        kind: lora::Kind,
    ) -> Result<(Self, lora::Artifact)> {
        Self::load_artifact_at(model, revision, device, path, kind, Precision::F32)
    }

    /// The same load at a stated precision.
    pub fn load_artifact_at(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        path: &Path,
        kind: lora::Kind,
        precision: Precision,
    ) -> Result<(Self, lora::Artifact)> {
        let base = BaseLoad::resolve(model, revision, device, precision)?;
        let artifact = lora::Artifact::load(path, &base.device)?;
        // A reward model's adapters exist to make its head separate, not to
        // change what the model writes; a generation adapter has no head to
        // score with. Either substitution answers a question nobody asked, and
        // does it plausibly, which is the reason it is refused rather than
        // reported.
        match (kind, artifact.kind) {
            (lora::Kind::Adapter, lora::Kind::Reward) => {
                bail!("adapter artifact is a reward model, not a generation adapter")
            }
            (lora::Kind::Reward, lora::Kind::Adapter) => {
                bail!("adapter artifact is a generation adapter, not a reward model")
            }
            _ => {}
        }
        if artifact.model != model {
            bail!(
                "adapter was trained for model {:?}, current model is {:?}",
                artifact.model,
                model
            );
        }
        if artifact.hidden_size != base.config.hidden_size {
            bail!(
                "adapter width {} does not match model width {}",
                artifact.hidden_size,
                base.config.hidden_size
            );
        }
        validate_layers(&artifact.layers, base.config.num_hidden_layers)?;
        base.architecture
            .check_targets(&artifact.targets, &base.config)?;
        artifact.validate_widths(lora::Widths::for_decoder(&base.config, &base.architecture))?;
        let adapters = lora::Adapters::from_artifact(&artifact, &base.device, base.dtype)?;
        let builder = base.builder()?;
        let model_impl = SteeringLlama::load_with_adapters(
            builder,
            base.config.clone(),
            base.architecture.clone(),
            adapters,
        )?;
        Ok((base.finish(model, model_impl), artifact))
    }

    /// Loads a model with fresh trainable adapters; the `VarMap` owns them.
    ///
    /// The base weights are mapped read-only exactly as `load` maps them and
    /// are never registered in the returned map, which is what makes "train
    /// only the adapters" a structural property rather than a convention: the
    /// optimizer is handed `varmap.all_vars()` and there is nothing else in it.
    pub fn load_trainable(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        spec: &lora::Spec,
    ) -> Result<(Self, VarMap)> {
        Self::load_trainable_at(model, revision, device, spec, Precision::F32)
    }

    /// The same load at a stated precision, and the one place the two dtypes
    /// differ.
    ///
    /// The base weights are mapped at `precision`; the adapters are created at
    /// `param_dtype`, which is always F32. That split is the whole of mixed
    /// precision and it is not decoration: a low-rank update is small relative
    /// to the weight it corrects, and in half precision an update below the
    /// weight's own ulp rounds to nothing — the adapter would train and the
    /// model would not move. `AdamW` builds both moments at each variable's
    /// own dtype, so keeping the variables in F32 keeps the optimizer state
    /// F32 for free, and `lora::Adapter::forward` already casts each factor to
    /// the activation's dtype through a differentiable `to_dtype`, whose
    /// backward casts the gradient back. Nothing between the two dtypes has to
    /// be arranged; it only has to not be undone.
    pub fn load_trainable_at(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        spec: &lora::Spec,
        precision: Precision,
    ) -> Result<(Self, VarMap)> {
        let base = BaseLoad::resolve(model, revision, device, precision)?;
        spec.validate(base.config.num_hidden_layers)?;
        base.architecture
            .check_targets(&spec.targets, &base.config)?;
        let spec = spec.resolved(base.config.num_hidden_layers);
        let widths = lora::Widths::for_decoder(&base.config, &base.architecture);
        let varmap = VarMap::new();
        let adapters =
            lora::Adapters::fresh(&spec, &varmap, widths, &base.device, base.param_dtype)?;
        let builder = base.builder()?;
        let model_impl = SteeringLlama::load_with_adapters(
            builder,
            base.config.clone(),
            base.architecture.clone(),
            adapters,
        )?;
        Ok((base.finish(model, model_impl), varmap))
    }
}
