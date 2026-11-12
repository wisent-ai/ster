//! The encoder pass and the head projection the T5 forward reads, and the
//! rows and lengths of a batch.

use super::*;

impl Model {
    pub(super) fn encode(&self, tokens: &Tensor, lengths: Option<&[usize]>) -> Result<EncoderState> {
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
                .map_err(|error| computing(&format!("running T5 encoder block {index}"), error))?;
        }
        let hidden = self.encoder_norm.forward(&hidden, Pass::Inference)?;
        let cross_attention = self
            .decoder
            .iter()
            .enumerate()
            .map(|(index, block)| {
                block.cross_memory(&hidden).map_err(|error| {
                    computing(
                        &format!("projecting the T5 encoder memory for decoder block {index}"),
                        error,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(EncoderState {
            hidden,
            cross_attention,
            mask,
        })
    }

    pub(super) fn project_head(&self, hidden: &Tensor) -> Result<Tensor> {
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

    pub(super) fn rows(
        &self,
        tokens: &Tensor,
        lengths: Option<&[usize]>,
        stack: &str,
        start: usize,
    ) -> Result<(usize, usize)> {
        let (batch, sequence) = tokens.dims2().map_err(|error| {
            computing(
                &format!("reading the T5 {stack} tokens' batch and sequence axes"),
                error,
            )
        })?;
        // Attention requires both a real example and at least one visible token.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        if batch == 0 || sequence == 0 {
            return Err(inconsistent(
                &t5::ROWS_EMPTY,
                stack,
                &format!(
                    "{batch} token rows of {sequence} tokens; a pass reads at least one token"
                ),
            ));
        }
        if let Some(lengths) = lengths.filter(|lengths| lengths.len() != batch) {
            return Err(inconsistent(
                &t5::LENGTHS_COUNT,
                stack,
                &format!("{} lengths for {batch} token rows", lengths.len()),
            ));
        }
        let Some(end) = start.checked_add(sequence) else {
            return Err(overflow(
                &format!("the {stack}'s positions"),
                &format!("position {start} plus {sequence} tokens"),
            ));
        };
        if let Some(limit) = self.config.context_length.filter(|limit| end > *limit) {
            return Err(refuse(
                &CONTEXT_EXCEEDED,
                &[
                    ("stack", stack),
                    ("end", &end.to_string()),
                    ("limit", &limit.to_string()),
                ],
            ));
        }
        Ok((batch, sequence))
    }
}
