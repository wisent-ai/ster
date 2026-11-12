//! One text read at one layer, every token's state kept: the read behind
//! `ster vector score`, which projects each onto a direction.

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::failures::{model::decoder::POSITIONS_UNKEPT, refuse, runtime::generate::UNSPELLABLE};

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
        let model = self.model.causal("ster vector score")?;
        let ids = self.encode_prompt(text)?;
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut cache = self.cache(model, false)?;
        let output = model.forward(&input, 0, &mut cache, None, &[layer])?;
        // `validate_layers(&[layer], ..)?` above refuses a layer past the
        // last, and `decode` runs `positions.insert(index, hidden.i(0)?)` for
        // every layer `capture_layers.binary_search(&index).is_ok()` names.
        let kept = output.positions.get(&layer);
        let rows = kept.map_or(Ok(0), |states| states.dim(0))?;
        let Some(states) = kept.filter(|_| rows == ids.len()) else {
            return Err(refuse(
                &POSITIONS_UNKEPT,
                &[
                    ("layer", &layer.to_string()),
                    ("kept", &rows.to_string()),
                    ("tokens", &ids.len().to_string()),
                ],
            ));
        };
        let states = states.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let texts = ids
            .iter()
            .map(|id| {
                self.tokenizer.decode(&[*id], false).map_err(|error| {
                    refuse(
                        &UNSPELLABLE,
                        &[
                            ("tokens", &format!("[{id}]")),
                            (
                                "vocabulary",
                                &self.tokenizer.get_vocab_size(true).to_string(),
                            ),
                            ("error", &error.to_string()),
                        ],
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TokenStates { ids, texts, states })
    }
}
