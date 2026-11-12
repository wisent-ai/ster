use std::{borrow::Cow, collections::BTreeMap};

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::Module;

use crate::failures::{
    model::t5::{self as t5, CONTEXT_EXCEEDED, inconsistent, overflow},
    refuse,
    tensor::computing,
};
use crate::model::{ForwardOutput, Mode, Pass, Readout, Route, SteeringPlan, input::Inputs};

use super::{Cache, Model, attention::mask, cache::EncoderState};

impl Model {
    pub fn forward(
        &self,
        inputs: Inputs<'_>,
        index_position: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        captures: &[usize],
        mode: Mode,
    ) -> Result<ForwardOutput> {
        let start = cache.position()?;
        if index_position != start {
            return Err(inconsistent(
                &t5::POSITION_MISMATCH,
                "decoder",
                &format!("given position {index_position}; its cache holds {start} positions"),
            ));
        }
        let (batch, queries) =
            self.rows(inputs.decoder, inputs.decoder_lengths, "decoder", start)?;
        if let Some(layers) = &cache.decoder {
            if layers.len() != self.decoder.len() {
                return Err(inconsistent(
                    &t5::CACHE_LAYERS_MISMATCH,
                    "decoder",
                    &format!(
                        "the cache holds {} layer slots for {} blocks",
                        layers.len(),
                        self.decoder.len()
                    ),
                ));
            }
            if inputs.decoder_lengths.is_some() || mode.pass == Pass::Differentiable {
                return Err(inconsistent(
                    &t5::CACHE_UNSUPPORTED,
                    "decoder",
                    &format!(
                        "a key-value cache with padded rows or a differentiable pass ({:?}); batched training and scoring run without one",
                        mode.pass
                    ),
                ));
            }
        }
        if inputs.decoder_lengths.is_some() && mode.readout == Readout::LastPosition {
            return Err(inconsistent(
                &t5::LAST_POSITION_PADDED,
                "decoder",
                "a padded batch asked for one common last position",
            ));
        }
        // A capture is one sequence's final-token state, never a padded batch slab.
        // https://ster.wisent.com/docs/concept-activation
        if !captures.is_empty() && (batch != 1 || inputs.decoder_lengths.is_some()) {
            return Err(inconsistent(
                &t5::CAPTURE_BATCHED,
                "decoder",
                &format!(
                    "captures {captures:?} asked of {batch} rows (padded: {}); a capture reads one unpadded sequence",
                    inputs.decoder_lengths.is_some()
                ),
            ));
        }
        if let Some(layer) = captures.iter().find(|layer| **layer >= self.decoder.len()) {
            return Err(inconsistent(
                &t5::CAPTURE_OUTSIDE,
                "decoder",
                &format!(
                    "capture layer {layer} of {} decoder blocks",
                    self.decoder.len()
                ),
            ));
        }
        if let Some(plan) = steering.filter(|plan| plan.hidden_size != self.config.hidden) {
            return Err(inconsistent(
                &t5::STEERING_WIDTH,
                "decoder",
                &format!(
                    "steering {} wide on a decoder {} wide",
                    plan.hidden_size, self.config.hidden
                ),
            ));
        }
        if inputs.encoder.is_none() && inputs.encoder_lengths.is_some() {
            return Err(inconsistent(
                &t5::LENGTHS_WITHOUT_TOKENS,
                "encoder",
                "lengths given without encoder token rows",
            ));
        }
        if let Some(tokens) = inputs.encoder {
            if cache.encoder.is_some() {
                return Err(inconsistent(
                    &t5::MEMORY_HELD,
                    "encoder",
                    "a source given to a cache that already holds encoder memory",
                ));
            }
            cache.encoder = Some(self.encode(tokens, inputs.encoder_lengths)?);
        }
        let Some(encoder) = cache.encoder.as_ref() else {
            return Err(inconsistent(
                &t5::MEMORY_MISSING,
                "decoder",
                "no source tokens and no encoder memory in the cache",
            ));
        };
        // Encoder and decoder batch axes must describe the same examples.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        let encoded = encoder
            .hidden
            .dim(0)
            .map_err(|error| computing("reading the T5 encoder memory's rows", error))?;
        if encoded != batch {
            return Err(inconsistent(
                &t5::ROWS_MISMATCH,
                "decoder",
                &format!("{encoded} encoded rows for {batch} decoder rows"),
            ));
        }
        let Some(keys) = start.checked_add(queries) else {
            return Err(overflow(
                "the decoder's keys",
                &format!("position {start} plus {queries} queries"),
            ));
        };
        let bias = self.decoder_bias.forward(queries, keys, start)?;
        let causal = mask::causal(
            inputs.decoder_lengths,
            queries,
            start,
            inputs.decoder.device(),
        )?;
        let captures = if captures.is_sorted() {
            Cow::Borrowed(captures)
        } else {
            let mut ordered = captures.to_vec();
            ordered.sort_unstable();
            Cow::Owned(ordered)
        };
        let mut activations = BTreeMap::new();
        let mut hidden = self.embeddings.forward(inputs.decoder)?;
        for (index, block) in self.decoder.iter().enumerate() {
            let slot = cache.decoder.as_mut().map(|layers| &mut layers[index]);
            hidden = block
                .forward(
                    &hidden,
                    &bias,
                    causal.as_ref(),
                    Some((&encoder.cross_attention[index], encoder.mask.as_ref())),
                    slot,
                    mode,
                )
                .map_err(|error| computing(&format!("running T5 decoder block {index}"), error))?;
            if captures.binary_search(&index).is_ok() {
                // Ster captures the last position after the block, before its intervention.
                // https://ster.wisent.com/docs/concept-activation
                let value = hidden
                    .i((0, queries - 1, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                activations.insert(index, value);
            }
            if let Some(plan) = steering {
                if let Some(vector) = plan.vector(index) {
                    let step = format!("adding the steering vector at T5 decoder block {index}");
                    hidden = hidden
                        .broadcast_add(&(vector * plan.strength)?)
                        .map_err(|error| computing(&step, error))?;
                }
            }
        }
        let hidden = self.decoder_norm.forward(&hidden, mode.pass)?;
        let logits = match mode.readout {
            Readout::Hidden => None,
            Readout::EveryPosition => Some(self.project_head(&hidden)?),
            Readout::LastPosition => {
                // Only the final query predicts the token generation is about to sample.
                // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
                Some(self.project_head(&hidden.i((.., queries - 1, ..))?)?)
            }
        };
        Ok(ForwardOutput {
            logits,
            hidden: hidden.to_dtype(DType::F32)?,
            activations,
            // A decoder that reads a target, not a text of its own: no
            // per-token read is offered here (`ster vector score` refuses it).
            positions: BTreeMap::new(),
        })
    }
}

mod encoded;
