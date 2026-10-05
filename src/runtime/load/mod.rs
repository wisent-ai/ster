//! Bringing a checkpoint into memory: the loaders an operator reaches, what
//! the three of them share, and the artifacts that may be attached on the way.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::llama::Config;
use tokenizers::Tokenizer;

use crate::{
    chat, lora,
    model::{Architecture, SteeringLlama},
};

use super::{DeviceChoice, Runtime, device::Precision, validate_layers};

mod checkpoint;
mod config;
mod family;
mod mistral;

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
        let model_impl = SteeringLlama::load(builder, base.config.clone(), base.architecture.clone())?;
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
        base.architecture.check_targets(&artifact.targets, &base.config)?;
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
        base.architecture.check_targets(&spec.targets, &base.config)?;
        let spec = spec.resolved(base.config.num_hidden_layers);
        let widths = projection_widths(&base.config, &base.architecture);
        let varmap = VarMap::new();
        let adapters = lora::Adapters::fresh(
            &spec,
            &varmap,
            widths,
            &base.device,
            base.param_dtype,
        )?;
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

/// Everything the three loaders share, held between resolving the checkpoint
/// and mapping its weights.
///
/// The split exists because `load_trainable` has to size its adapters from the
/// config *before* the base weights are mapped, and because three copies of
/// the config parsing, the architecture refusal, and the tokenizer load would
/// drift the moment one of them gained a check the other two did not.
struct BaseLoad {
    tokenizer: Tokenizer,
    config: Config,
    architecture: Architecture,
    weights: Vec<PathBuf>,
    layout: Layout,
    revision: Option<String>,
    eos_tokens: BTreeSet<u32>,
    chat: std::result::Result<Option<chat::Template>, String>,
    device: Device,
    dtype: DType,
    precision: Precision,
    param_dtype: DType,
}

impl BaseLoad {
    fn resolve(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        precision: Precision,
    ) -> Result<Self> {
        let device = device.resolve()?;
        let dtype = precision.dtype(&device)?;
        // Never the base dtype. Everything this run creates and steps stays in
        // single precision whatever the frozen weights are mapped at.
        let param_dtype = DType::F32;
        let source = Checkpoint::resolve(model, revision)?;
        let (config, architecture, eos_tokens) = source.decoder_config()?;
        // A template that does not compile is kept as its refusal, raised
        // only by a run that asks for the template.
        let chat = source.chat().map_err(|error| format!("{error:#}"));
        let tokenizer = super::vocabulary::load(&source.tokenizer, source.tokenizer_config.as_deref())?;
        Ok(Self {
            tokenizer,
            config,
            architecture,
            weights: source.weights,
            layout: source.layout,
            revision: source.revision,
            eos_tokens,
            chat,
            device,
            dtype,
            precision,
            param_dtype,
        })
    }

    /// Maps the base weights read-only. Nothing here is registered in a
    /// `VarMap`, so the base stays frozen whichever loader called it. A
    /// Mistral-format checkpoint answers each Transformers name from its own.
    fn builder(&self) -> Result<VarBuilder<'static>> {
        let builder = unsafe { VarBuilder::from_mmaped_safetensors(&self.weights, self.dtype, &self.device) }
            .with_context(|| format!("failed to map {} model weight files", self.weights.len()))?;
        let layout = self.layout;
        Ok(match layout {
            Layout::Transformers => builder,
            Layout::Mistral => builder.rename_f(move |name: &str| layout.stored_name(name)),
        })
    }

    fn finish(self, model_id: &str, model: SteeringLlama) -> Runtime {
        Runtime {
            model_id: model_id.to_owned(),
            revision: self.revision,
            tokenizer: self.tokenizer,
            model,
            device: self.device,
            dtype: self.dtype,
            precision: self.precision,
            param_dtype: self.param_dtype,
            eos_tokens: self.eos_tokens,
            chat: self.chat,
            chat_status: chat::Status::Off,
        }
    }
}

/// The width of every projection an adapter can sit on.
///
/// Grouped-query attention gives the key and value projections fewer heads
/// than the query projection, so their output is `num_key_value_heads *
/// head_dim` wide rather than `hidden_size` wide, and a config that states
/// `head_dim` can make the query and output projections differ from the
/// residual width too. An adapter sized from `hidden_size` would fail to
/// matmul against them.
pub(super) fn projection_widths(config: &Config, architecture: &Architecture) -> lora::Widths {
    lora::Widths {
        hidden: config.hidden_size,
        attention: architecture.attention_width(config.num_attention_heads),
        attention_output: config.num_attention_heads * architecture.value_dim(),
        key_value: config.num_key_value_heads * architecture.head_dim,
        intermediate: config.intermediate_size,
    }
}
