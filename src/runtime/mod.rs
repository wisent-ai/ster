//! One loaded checkpoint and everything a caller does with it.
//!
//! `load` brings the files into memory, `text` turns prompts into the exact
//! tokens this checkpoint expects, `infer` runs the model and samples from
//! it, and `device` names where and in what precision all of that happens.
//! What stays here is the handle itself and what it can be asked about
//! without running anything.

use std::collections::BTreeSet;

use anyhow::Result;
use candle_core::{DType, Device};
use tokenizers::Tokenizer;

use crate::{chat, model::Network};

mod artifact;
mod device;
mod infer;
mod load;
mod text;
pub(crate) mod vocabulary;

pub use device::{DeviceChoice, Precision};
pub use infer::{Completion, GenerationOptions};
pub use load::{Checkpoint, Layout};

pub struct Runtime {
    pub model_id: String,
    pub revision: Option<String>,
    tokenizer: Tokenizer,
    model: Network,
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
    chat: std::result::Result<Option<chat::Template>, anyhow::Error>,
    chat_status: chat::Status,
}

impl Runtime {
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
        self.model.hidden_size()
    }

    /// The layers steering, capture and adapters address: a T5 model's
    /// decoder blocks.
    pub fn layer_count(&self) -> usize {
        self.model.layer_count()
    }

    /// The rows of the model's embedding table: every token id it reads.
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// The longest sequence the model was configured for
    /// (`max_position_embeddings`, T5's `n_positions`). Training clamps
    /// against it for the same reason `generate` does: past it the model's
    /// positions are untrained.
    pub fn context_length(&self) -> Result<usize> {
        self.model.context_length()
    }
}

/// Refuses the first of `layers` a model of `count` layers lacks.
pub(super) fn validate_layers(layers: &[usize], count: usize) -> Result<()> {
    if let Some(layer) = layers.iter().copied().find(|layer| *layer >= count) {
        let named: Vec<String> = layers.iter().map(usize::to_string).collect();
        return Err(crate::failures::refuse(
            &crate::failures::runtime::steer::LAYER_OUTSIDE,
            &[
                ("layer", &layer.to_string()),
                ("count", &count.to_string()),
                ("layers", &named.join(", ")),
                ("last", &count.saturating_sub(1).to_string()),
            ],
        ));
    }
    Ok(())
}

/// The activations one forward pass returned, in layer order. The decoder
/// keeps the last position's state at exactly the layers it is asked for,
/// so a read that came back with another set, or an empty one, is refused
/// as a defect of the decoder, not of anything a run was given.
pub(super) fn requested_activations(
    read: std::collections::BTreeMap<usize, Vec<f32>>,
    layers: &[usize],
) -> Result<Vec<(usize, Vec<f32>)>> {
    let requested: BTreeSet<usize> = layers.iter().copied().collect();
    let returned: BTreeSet<usize> = read.keys().copied().collect();
    let empty: Vec<usize> = read
        .iter()
        .filter(|(_, values)| values.is_empty())
        .map(|(layer, _)| *layer)
        .collect();
    if requested != returned || !empty.is_empty() {
        return Err(crate::failures::refuse(
            &crate::failures::model::decoder::ACTIVATIONS_UNREQUESTED,
            &[
                ("requested", &format!("{requested:?}")),
                ("returned", &format!("{returned:?}")),
                ("empty", &format!("{empty:?}")),
            ],
        ));
    }
    Ok(read.into_iter().collect())
}
