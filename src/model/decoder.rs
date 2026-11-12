//! The whole decoder: embeddings, the stack of blocks, the final norm and
//! the head — plus the one entry every caller runs a forward pass through.

mod decode;
mod embeddings;
mod load;
mod stack;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Embedding, Linear};
use candle_transformers::models::llama::Config;

use crate::failures::lifted;
use crate::lora::Adapters;

use super::{
    Architecture, Cache, ForwardOutput, Mode, Readout, SteeringPlan,
    attention::attend::padded_causal_mask,
    deepseek4::HyperHead,
    depth::DepthMix,
    layer::{DecoderLayer, altup::StreamProjections, norm::Norm},
};
use embeddings::{NgramEmbeddings, PerLayerEmbeddings};

#[derive(Debug, Clone)]
pub struct SteeringLlama {
    embeddings: Embedding,
    /// The learned position table and how many rows it keeps before
    /// position zero (GPT-2, OPT, GPT-BigCode).
    positions: Option<(Embedding, usize)>,
    /// BLOOM's norm straight after the embedding.
    embedding_norm: Option<Norm>,
    layers: Vec<DecoderLayer>,
    /// Gemma 4's per-layer input tables.
    per_layer: Option<PerLayerEmbeddings>,
    /// LongCat-Flash-Lite's n-gram tables.
    ngram: Option<NgramEmbeddings>,
    /// Gemma 3n's projections into and out of its AltUp streams.
    streams: Option<StreamProjections>,
    /// HRM-Text's starting low state (`model.z_L_init`).
    low_start: Option<Tensor>,
    /// Kimi-K3's attention-residual mix before the final norm.
    depth_output: Option<DepthMix>,
    /// DeepSeek-V4's collapse of its residual streams before the final norm.
    hyper_head: Option<HyperHead>,
    final_norm: Norm,
    lm_head: Linear,
    config: Config,
    architecture: Architecture,
    adapters: Adapters,
}

impl SteeringLlama {
    pub fn adapters(&self) -> &crate::lora::Adapters {
        &self.adapters
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }

    pub fn forward(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
    ) -> Result<ForwardOutput> {
        self.forward_pass(
            tokens,
            index_pos,
            cache,
            steering,
            capture_layers,
            Mode::DECODE,
        )
    }

    /// `mode` picks the kernels, the adapter route and the readout; every
    /// other argument is unchanged. A refusal a layer declared comes out as
    /// that refusal.
    pub fn forward_pass(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
        mode: Mode,
    ) -> Result<ForwardOutput> {
        self.decode(
            tokens,
            index_pos,
            cache,
            steering,
            capture_layers,
            None,
            mode,
        )
        .map_err(lifted)
    }

    /// A batch of unequal-length sequences in a single pass.
    ///
    /// `tokens` is `[batch, sequence]` and `lengths` says how many of each
    /// row's columns are real; everything past a row's length is filler the
    /// caller stacked to make the rectangle. The logits come back
    /// `[batch, sequence, vocab]` — a full rectangle, of which only the first
    /// `lengths[row]` rows of each slab mean anything.
    ///
    /// **Padding sits on the right**, and both consequences are load-bearing.
    /// The first is positional: with no key-value cache every column `j` is
    /// absolute position `j`, so right padding leaves every real token at the
    /// position it would have held alone, and the one shared rotation of
    /// positions `0..sequence` is correct for all rows at once. Left padding would
    /// shift each row's real tokens by its own pad count, which no scalar
    /// `index_pos` can express and which this decoder has no per-row position
    /// argument to carry. The second is what the caller must then do: a row's
    /// real logits are rows `0..lengths[row]` of its slab, so a loss slices
    /// each row from the front and stops at its own length rather than
    /// slicing a common window off the end. (Left padding is the right choice
    /// for batched *decoding*, where aligning every row's last real token at
    /// the final column is what lets one cached step serve the whole batch.
    /// This path has no cache and never decodes.)
    ///
    /// Refusals, rather than a plausible-looking loss over filler: a batch
    /// with no rows or no columns, a `lengths` that does not describe every
    /// row, a row claiming more tokens than the batch is wide, a row claiming
    /// none at all, a batch wider than the positions the model was built for, a readout of one
    /// last position, and a key-value cache.
    ///
    /// Activations are not captured here. Capture is defined as row zero's
    /// final column, which in a padded batch is filler, and a per-row capture
    /// is a different feature than the one the steering path asked for.
    pub fn forward_batch(
        &self,
        tokens: &Tensor,
        lengths: &[usize],
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        mode: Mode,
    ) -> Result<ForwardOutput> {
        let (batch, sequence) = tokens.dims2()?;
        // Every caller is `Runtime::forward_rows`, which pads encoded rows
        // (each of at least one token) to the longest, never past the limit
        // its run clamped to `max_position_embeddings`, with no cache and a
        // readout of every position: each of these is a defect of a caller,
        // not of anything a run was given.
        assert!(
            batch > 0 && sequence > 0,
            "a batched forward needs at least one row of at least one token"
        );
        assert_eq!(
            lengths.len(),
            batch,
            "a batched forward gets one length per row"
        );
        assert!(
            !cache.use_kv_cache,
            "a batched forward cannot share one key-value cache across rows"
        );
        assert!(
            sequence <= self.config.max_position_embeddings,
            "a batch {sequence} tokens wide exceeds the {} positions this model was built for",
            self.config.max_position_embeddings
        );
        assert!(
            !matches!(mode.readout, Readout::LastPosition),
            "a batched forward cannot read one last position, because every row ends somewhere else"
        );
        assert!(
            lengths
                .iter()
                .all(|&length| length > 0 && length <= sequence),
            "every row of a batch holds tokens and fits its width"
        );
        let full = padded_causal_mask(lengths, sequence, None, None, tokens.device())?;
        let windowed = match self.architecture.sliding_window {
            Some(window) if self.architecture.sliding_layers != 0 => Some(padded_causal_mask(
                lengths,
                sequence,
                Some(window),
                self.architecture.chunk_lookback,
                tokens.device(),
            )?),
            _ => None,
        };
        let masks = BatchMasks { full, windowed };
        self.decode(tokens, 0, cache, steering, &[], Some(&masks), mode)
            .map_err(lifted)
    }
}

impl SteeringLlama {
    /// Cohere multiplies its final logits by `logit_scale`, Granite divides
    /// them by `logits_scaling` and Inkling by `logits_mup_width_multiplier`;
    /// Gemma 2 and 3 bound them to `(-cap, cap)` with a tanh; Inkling scores
    /// only the first `unpadded_vocab_size` tokens.
    fn soft_cap(&self, logits: Tensor) -> candle_core::Result<Tensor> {
        let logits = match self.architecture.logits_multiplier {
            Some(multiplier) => (logits * multiplier)?,
            None => logits,
        };
        let logits = match self.architecture.vocabulary_limit {
            Some(limit) if limit < logits.dim(candle_core::D::Minus1)? => logits
                .narrow(candle_core::D::Minus1, 0, limit)?
                .contiguous()?,
            _ => logits,
        };
        match self.architecture.final_softcap {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }
    }
}

/// The padded causal masks one batch needs: full attention, and the sliding
/// window when the model has local layers.
pub(super) struct BatchMasks {
    full: Tensor,
    windowed: Option<Tensor>,
}

impl BatchMasks {
    fn for_window(&self, window: Option<usize>) -> &Tensor {
        match (window, &self.windowed) {
            (Some(_), Some(windowed)) => windowed,
            _ => &self.full,
        }
    }
}
