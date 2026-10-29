//! The network one loaded checkpoint runs: a causal decoder, which reads a
//! prompt and its continuation as one token sequence, or T5's encoder and
//! decoder, which read the prompt in the encoder and write the answer in the
//! decoder. Layer numbers, steering vectors and captures address the decoder
//! stack of either, so a direction's layer means the same thing in both.

use anyhow::{Result, bail};

use crate::lora::Adapters;

use super::{SteeringLlama, t5};

#[derive(Debug)]
pub(crate) enum Network {
    Causal(SteeringLlama),
    EncoderDecoder(t5::Model),
}

impl Network {
    /// The residual width every steering vector and capture has.
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Causal(model) => model.config().hidden_size,
            Self::EncoderDecoder(model) => model.config().hidden,
        }
    }

    /// The layers steering, capture and adapters address: T5's decoder
    /// blocks, since its encoder only reads the prompt.
    pub fn layer_count(&self) -> usize {
        match self {
            Self::Causal(model) => model.config().num_hidden_layers,
            Self::EncoderDecoder(model) => model.config().decoder_layers,
        }
    }

    /// The rows of the embedding table: every token id the model reads.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Causal(model) => model.config().vocab_size,
            Self::EncoderDecoder(model) => model.config().vocab,
        }
    }

    /// The longest sequence the model was configured for: a causal model's
    /// `max_position_embeddings`, a T5 model's `n_positions`, which its
    /// config.json states as an extra key `T5Config` itself does not define.
    pub fn context_length(&self) -> Result<usize> {
        match self {
            Self::Causal(model) => Ok(model.config().max_position_embeddings),
            Self::EncoderDecoder(model) => match model.config().context_length {
                Some(length) => Ok(length),
                None => bail!(
                    "this T5 checkpoint's config.json states no n_positions, so Ster has no sequence length to bound its prompts by; add n_positions, the length the checkpoint was trained at, to its config.json"
                ),
            },
        }
    }

    pub fn adapters(&self) -> &Adapters {
        match self {
            Self::Causal(model) => model.adapters(),
            Self::EncoderDecoder(model) => model.adapters(),
        }
    }

    /// The causal decoder, or the refusal of `command`, a pass that scores
    /// one token sequence from its first position, on a T5 checkpoint.
    pub fn causal(&self, command: &str) -> Result<&SteeringLlama> {
        match self {
            Self::Causal(model) => Ok(model),
            Self::EncoderDecoder(_) => bail!(
                "{command} cannot run this t5 checkpoint: it scores one token sequence from its first position, and a T5 model reads the prompt in its encoder and writes the answer in its decoder, a split that pass does not carry; run {command} on a causal-decoder checkpoint"
            ),
        }
    }
}
