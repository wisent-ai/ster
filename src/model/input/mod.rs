//! Borrowed token tensors for one encoder-decoder pass. Encoder tokens never
//! become decoder history; a causal model reads its one sequence directly.

use candle_core::Tensor;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Inputs<'a> {
    /// Supplied when starting an encoder-decoder pass, omitted when its cache
    /// already holds the encoder result.
    pub encoder: Option<&'a Tensor>,
    pub encoder_lengths: Option<&'a [usize]>,
    pub decoder: &'a Tensor,
    pub decoder_lengths: Option<&'a [usize]>,
}
