//! One text read at one layer, every token's state kept: the read behind
//! `ster vector score`, which projects each onto a direction.

use anyhow::{Result, bail};
use candle_core::{DType, Tensor};

use crate::model::Network;

use super::super::{Runtime, validate_layers};

/// A text's tokens as the model read them, and each one's state at a layer.
pub struct TokenStates {
    /// Every token id the model was given, in order.
    pub ids: Vec<u32>,
    /// Each id as the tokenizer spells it alone.
    pub texts: Vec<String>,
    /// `[tokens, hidden]`, F32, one row per id.
    pub states: Vec<Vec<f32>>,
}

impl Runtime {
    /// `text` read as a prompt is read (`encode_prompt`, the chat template
    /// applied as this run decided), and every position's state at `layer`,
    /// taken after the layer as `activations` takes the last one.
    pub fn token_states(&self, text: &str, layer: usize) -> Result<TokenStates> {
        validate_layers(&[layer], self.layer_count())?;
        let model = match &self.model {
            Network::Causal(model) => model,
            Network::EncoderDecoder(_) => bail!(
                "an encoder-decoder checkpoint reads a text into its encoder and decodes a target of its own, so it has no per-token states of the text to score; use a decoder-only model"
            ),
        };
        let ids = self.encode_prompt(text)?;
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut cache = self.cache(model, false)?;
        let output = model.forward(&input, 0, &mut cache, None, &[layer])?;
        let Some(states) = output.positions.get(&layer) else {
            bail!("the forward pass kept no states at layer {layer}");
        };
        let states = states.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        if states.len() != ids.len() {
            bail!("the forward pass kept {} states for {} tokens at layer {layer}", states.len(), ids.len());
        }
        let texts = ids
            .iter()
            .map(|id| {
                self.tokenizer
                    .decode(&[*id], false)
                    .map_err(|error| anyhow::anyhow!("token {id} cannot be spelled by the tokenizer: {error}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TokenStates { ids, texts, states })
    }
}
