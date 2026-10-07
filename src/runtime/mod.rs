//! One loaded checkpoint and everything a caller does with it.
//!
//! `load` brings the files into memory, `text` turns prompts into the exact
//! tokens this checkpoint expects, `infer` runs the model and samples from
//! it, and `device` names where and in what precision all of that happens.
//! What stays here is the handle itself and what it can be asked about
//! without running anything.

use std::collections::BTreeSet;

use anyhow::{bail, Result};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;

use crate::{chat, model::SteeringLlama};

mod artifact;
mod device;
mod infer;
mod load;
mod text;
mod vocabulary;

pub use device::{DeviceChoice, Precision};
pub use infer::{Completion, GenerationOptions};
pub use load::{Checkpoint, Layout};

use load::projection_widths;

pub struct Runtime {
    pub model_id: String,
    pub revision: Option<String>,
    tokenizer: Tokenizer,
    model: SteeringLlama,
    device: Device,
    dtype: DType,
    /// The flag value that chose `dtype`, kept alongside it because a written
    /// artifact records the operator's word ("f16") rather than Candle's
    /// enum, and deriving one from the other would have to guess which of
    /// several spellings produced a given `DType`.
    precision: Precision,
    /// The dtype every weight this run creates is held in, which is always
    /// F32 whatever the base is mapped at. See `load_trainable_at`.
    param_dtype: DType,
    eos_tokens: BTreeSet<u32>,
    /// The conversation format the checkpoint publishes, compiled at load
    /// time — or why it does not compile, refused only when a run asks for
    /// it — and whether this run uses it.
    ///
    /// The choice lives on the runtime rather than in each objective's
    /// options because the encoders are what need it, every objective reaches
    /// them through this one handle, and a per-objective copy of the flag
    /// would let supervised fine-tuning and preference optimization train the
    /// same checkpoint in two different shapes. It starts `Off`, so every
    /// path that never asks — steering, extraction, synthesis — encodes
    /// exactly the text it encoded before this existed.
    chat: std::result::Result<Option<chat::Template>, String>,
    chat_status: chat::Status,
}

impl Runtime {

    /// The projection widths an adapter has to match: the residual width,
    /// the attention width, one key or value projection, and the feed
    /// forward. Grouped-query attention makes the key and value width smaller
    /// than the attention width, and a stated `head_dim` can make the
    /// attention width differ from the residual one, which is why none of
    /// them can be derived from `hidden_size`.
    pub fn config_dims(&self) -> crate::lora::Widths {
        projection_widths(self.model.config(), self.model.architecture())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The dtype the frozen base weights are mapped at, and the dtype every
    /// activation flowing through them carries.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The precision flag this run was loaded at, as the operator spelled it.
    /// What a written artifact records, so a later run can be told it is
    /// reading a direction fitted in a space it is not measuring in.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The dtype a weight this run creates and steps must be held in: always
    /// F32. A reward head registered at the base dtype would be a trained
    /// parameter in half precision, which is the one thing mixed precision
    /// exists to avoid.
    pub fn param_dtype(&self) -> DType {
        self.param_dtype
    }

    pub fn hidden_size(&self) -> usize {
        self.model.config().hidden_size
    }

    pub fn layer_count(&self) -> usize {
        self.model.config().num_hidden_layers
    }

    /// The rows of the model's embedding table: every token id it reads.
    pub fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    /// The longest sequence the model was configured for
    /// (`max_position_embeddings`). Training clamps against it for the same
    /// reason `generate` does: past it the model's positions are untrained.
    pub fn context_length(&self) -> usize {
        self.model.config().max_position_embeddings
    }


}

pub(super) fn validate_layers(layers: &[usize], count: usize) -> Result<()> {
    if layers.is_empty() {
        bail!("at least one layer is required");
    }
    if let Some(layer) = layers.iter().copied().find(|layer| *layer >= count) {
        bail!("layer {layer} is outside the model's 0..{} range", count.saturating_sub(1));
    }
    Ok(())
}

