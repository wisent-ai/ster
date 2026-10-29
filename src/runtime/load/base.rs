//! What the three loaders share: the checkpoint resolved, its config parsed and
//! its weights mapped.

use std::{collections::BTreeSet, path::PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::Config;
use tokenizers::Tokenizer;

use crate::{
    chat,
    model::{Architecture, Network, SteeringLlama, StoredNames},
    runtime::{DeviceChoice, Runtime, device::Precision},
};

use super::{Checkpoint, Layout, gguf, hy_v4};

/// Everything the three loaders share, held between resolving the checkpoint
/// and mapping its weights.
///
/// The split exists because `load_trainable` has to size its adapters from the
/// config *before* the base weights are mapped, and because three copies of
/// the config parsing, the architecture refusal, and the tokenizer load would
/// drift the moment one of them gained a check the other two did not.
pub(super) struct BaseLoad {
    pub(super) tokenizer: Tokenizer,
    pub(super) config: Config,
    pub(super) architecture: Architecture,
    pub(super) weights: Vec<PathBuf>,
    pub(super) layout: Layout,
    pub(super) revision: Option<String>,
    pub(super) eos_tokens: BTreeSet<u32>,
    pub(super) chat: std::result::Result<Option<chat::Template>, String>,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) precision: Precision,
    pub(super) param_dtype: DType,
}

impl BaseLoad {
    pub(super) fn resolve(
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
        let tokenizer = crate::runtime::vocabulary::load(
            &source.tokenizer,
            source.tokenizer_config.as_deref(),
        )?;
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
    /// Mistral-format checkpoint answers each Transformers name from its own,
    /// and so do an HY-V4 checkpoint under its release names and a
    /// DeepSeek-V4 checkpoint in DeepSeek's own layout (`embed`,
    /// `layers.{i}.attn.wq_a`, `ffn.experts`).
    pub(super) fn builder(&self) -> Result<VarBuilder<'static>> {
        let layout = self.layout;
        let builder = match layout {
            // One GGUF file: each tensor is dequantized at this run's dtype
            // under the Transformers name the decoder asks for.
            Layout::Gguf => {
                let [file] = self.weights.as_slice() else {
                    bail!(
                        "a GGUF checkpoint is one .gguf file; this one lists {}",
                        self.weights.len()
                    );
                };
                VarBuilder::from_backend(
                    Box::new(gguf::GgufWeights::open(file)?),
                    self.dtype,
                    self.device.clone(),
                )
            }
            Layout::Transformers | Layout::Mistral => {
                let mapped = unsafe {
                    VarBuilder::from_mmaped_safetensors(&self.weights, self.dtype, &self.device)
                }
                .with_context(|| {
                    format!("failed to map {} model weight files", self.weights.len())
                })?;
                match (layout, self.architecture.stored_names) {
                    (Layout::Mistral, _) => {
                        mapped.rename_f(move |name: &str| layout.stored_name(name))
                    }
                    (_, StoredNames::HyV4) => {
                        let names = hy_v4::StoredNames::read(&self.weights)?;
                        mapped.rename_f(move |name: &str| names.stored(name))
                    }
                    (_, StoredNames::Transformers) => mapped,
                }
            }
        };
        if self.architecture.compressed.is_none() {
            return Ok(builder);
        }
        let root = ["", "model."]
            .into_iter()
            .find(|root| builder.contains_tensor(&format!("{root}embed.weight")));
        Ok(match root {
            Some(root) => {
                let key_value_norm =
                    if builder.contains_tensor(&format!("{root}layers.0.attn.kv_norm.weight")) {
                        "kv_norm"
                    } else {
                        "norm"
                    };
                let names = crate::model::NativeNames {
                    root: root.to_owned(),
                    key_value_norm,
                };
                builder.rename_f(move |name: &str| names.stored(name))
            }
            None => builder,
        })
    }

    pub(super) fn finish(self, model_id: &str, model: SteeringLlama) -> Runtime {
        Runtime {
            model_id: model_id.to_owned(),
            revision: self.revision,
            tokenizer: self.tokenizer,
            model: Network::Causal(model),
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
