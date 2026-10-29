use std::{borrow::Cow, collections::BTreeMap};

use anyhow::{Context, Result, bail};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::Module;

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
            bail!(
                "T5 decoder was given position {index_position}, but its cache holds {start} positions"
            );
        }
        let (batch, queries) =
            self.rows(inputs.decoder, inputs.decoder_lengths, "decoder", start)?;
        if let Some(layers) = &cache.decoder {
            if layers.len() != self.decoder.len() {
                bail!(
                    "T5 decoder cache holds {} layer slots for {} blocks",
                    layers.len(),
                    self.decoder.len()
                );
            }
            if inputs.decoder_lengths.is_some() || mode.pass == Pass::Differentiable {
                bail!(
                    "T5 batched training and scoring require a pass without a decoder key-value cache"
                );
            }
        }
        if inputs.decoder_lengths.is_some() && mode.readout == Readout::LastPosition {
            bail!("a padded T5 batch cannot read one common last position");
        }
        // A capture is one sequence's final-token state, never a padded batch slab.
        // https://ster.wisent.com/docs/concept-activation
        if !captures.is_empty() && (batch != 1 || inputs.decoder_lengths.is_some()) {
            bail!("T5 activation capture requires one unpadded decoder sequence");
        }
        if let Some(layer) = captures.iter().find(|layer| **layer >= self.decoder.len()) {
            bail!(
                "T5 capture layer {layer} is outside its {} decoder blocks",
                self.decoder.len()
            );
        }
        if steering.is_some_and(|plan| plan.hidden_size != self.config.hidden) {
            bail!(
                "T5 steering width does not match decoder hidden width {}",
                self.config.hidden
            );
        }
        if inputs.encoder.is_none() && inputs.encoder_lengths.is_some() {
            bail!("T5 encoder lengths were supplied without encoder token rows");
        }
        if let Some(tokens) = inputs.encoder {
            if cache.encoder.is_some() {
                bail!(
                    "T5 cache already contains encoder memory; start a new pass before replacing its source"
                );
            }
            cache.encoder = Some(self.encode(tokens, inputs.encoder_lengths)?);
        }
        let encoder = cache
            .encoder
            .as_ref()
            .context("T5 decoder requires source tokens or previously encoded source memory")?;
        // Encoder and decoder batch axes must describe the same examples.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        if encoder.hidden.dim(0)? != batch {
            bail!("T5 encoder and decoder batch sizes differ");
        }
        let keys = start
            .checked_add(queries)
            .context("T5 decoder position overflow")?;
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
                .with_context(|| format!("failed in T5 decoder block {index}"))?;
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
                    hidden = hidden
                        .broadcast_add(&(vector * plan.strength)?)
                        .with_context(|| {
                            format!("failed to apply T5 decoder steering at layer {index}")
                        })?;
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
        })
    }

    fn encode(&self, tokens: &Tensor, lengths: Option<&[usize]>) -> Result<EncoderState> {
        // Encoder positions always begin at the source sequence origin.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        let (_, sequence) = self.rows(tokens, lengths, "encoder", 0)?;
        let mask = lengths
            .map(|lengths| mask::padding(lengths, sequence, tokens.device()))
            .transpose()?
            .flatten();
        // Encoder attention is bidirectional and has no decoder prefix.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        let bias = self.encoder_bias.forward(sequence, sequence, 0)?;
        let mode = Mode {
            pass: Pass::Inference,
            route: Route::Base,
            readout: Readout::Hidden,
        };
        let mut hidden = self.embeddings.forward(tokens)?;
        for (index, block) in self.encoder.iter().enumerate() {
            hidden = block
                .forward(&hidden, &bias, mask.as_ref(), None, None, mode)
                .with_context(|| format!("failed in T5 encoder block {index}"))?;
        }
        let hidden = self.encoder_norm.forward(&hidden, Pass::Inference)?;
        let cross_attention = self
            .decoder
            .iter()
            .enumerate()
            .map(|(index, block)| {
                block.cross_memory(&hidden).with_context(|| {
                    format!("failed to project T5 encoder memory for decoder block {index}")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(EncoderState {
            hidden,
            cross_attention,
            mask,
        })
    }

    fn project_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = if self.config.scale_decoder_outputs {
            // Original T5 scales tied-head input; FLAN's declared untied head does not.
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            (hidden * (self.config.hidden as f64).powf(-0.5))?
        } else {
            hidden.clone()
        };
        Ok(self
            .head
            .forward(&hidden.contiguous()?)?
            .to_dtype(DType::F32)?)
    }

    fn rows(
        &self,
        tokens: &Tensor,
        lengths: Option<&[usize]>,
        stack: &str,
        start: usize,
    ) -> Result<(usize, usize)> {
        let (batch, sequence) = tokens
            .dims2()
            .with_context(|| format!("T5 {stack} tokens must have batch and sequence axes"))?;
        // Attention requires both a real example and at least one visible token.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        if batch == 0 || sequence == 0 {
            bail!("T5 {stack} input requires nonempty token rows");
        }
        if lengths.is_some_and(|lengths| lengths.len() != batch) {
            bail!("T5 {stack} lengths do not describe its {batch} token rows");
        }
        let end = start
            .checked_add(sequence)
            .context("T5 sequence position overflow")?;
        if let Some(limit) = self.config.context_length {
            if end > limit {
                bail!(
                    "T5 {stack} sequence reaches position {end}, beyond its declared n_positions={limit}"
                );
            }
        }
        Ok((batch, sequence))
    }
}
