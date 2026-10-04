//! The whole decoder: embeddings, the stack of blocks, the final norm and
//! the head — plus the one entry every caller runs a forward pass through.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias};
use candle_transformers::models::llama::Config;

use crate::lora::Adapters;

use super::{
    Architecture, Cache, ForwardOutput, Mode, Readout, SteeringPlan,
    attention::padded_causal_mask,
    layer::{DecoderLayer, load_norm, normalize},
};

#[derive(Debug, Clone)]
pub struct SteeringLlama {
    embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    config: Config,
    architecture: Architecture,
    adapters: Adapters,
}

impl SteeringLlama {
    pub fn load(
        builder: VarBuilder<'_>,
        config: Config,
        architecture: Architecture,
    ) -> Result<Self> {
        Ok(Self::load_with_adapters(
            builder,
            config,
            architecture,
            Adapters::default(),
        )?)
    }

    /// Loads the frozen base and attaches `adapters`.
    ///
    /// The base weights come from `builder`, which Ster maps read-only out of
    /// safetensors; only the adapter factors were ever registered in a `VarMap`.
    /// Attaching them here therefore cannot make a base weight trainable, which
    /// is the property the whole training path rests on.
    pub fn load_with_adapters(
        builder: VarBuilder<'_>,
        config: Config,
        architecture: Architecture,
        adapters: crate::lora::Adapters,
    ) -> candle_core::Result<Self> {
        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            builder.pp("model.embed_tokens"),
        )?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, builder.pp("lm_head"))?
        };
        let final_norm = load_norm(
            config.hidden_size,
            config.rms_norm_eps,
            architecture.norm_offset,
            builder.pp("model.norm"),
        )?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                DecoderLayer::load(
                    builder.pp(format!("model.layers.{index}")),
                    &config,
                    architecture,
                    index,
                    &adapters,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            embeddings,
            layers,
            final_norm,
            lm_head,
            config,
            architecture,
            adapters,
        })
    }

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
        Ok(self.forward_pass(
            tokens,
            index_pos,
            cache,
            steering,
            capture_layers,
            Mode::DECODE,
        )?)
    }

    /// `mode` picks the kernels, the adapter route and the readout; every
    /// other argument is unchanged.
    pub fn forward_pass(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
        mode: Mode,
    ) -> candle_core::Result<ForwardOutput> {
        self.decode(
            tokens,
            index_pos,
            cache,
            steering,
            capture_layers,
            None,
            mode,
        )
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
    /// position it would have held alone, and the one shared rotary window
    /// `cos[0..sequence]` is correct for all rows at once. Left padding would
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
    /// none at all, a batch wider than the rotary tables, a readout of one
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
        if batch == 0 || sequence == 0 {
            bail!("a batched forward needs at least one row of at least one token");
        }
        if lengths.len() != batch {
            bail!(
                "a batched forward got {} lengths for {batch} rows",
                lengths.len()
            );
        }
        if cache.use_kv_cache {
            bail!("a batched forward cannot share one key-value cache across rows");
        }
        if sequence > self.config.max_position_embeddings {
            bail!(
                "a batch {sequence} tokens wide exceeds the {} positions this model was built for",
                self.config.max_position_embeddings
            );
        }
        if matches!(mode.readout, Readout::LastPosition) {
            bail!(
                "a batched forward cannot read one last position, because every row ends somewhere else"
            );
        }
        for (row, &length) in lengths.iter().enumerate() {
            if length == 0 {
                bail!("row {row} of the batch holds no tokens");
            }
            if length > sequence {
                bail!("row {row} claims {length} tokens in a batch only {sequence} wide");
            }
        }
        let full = padded_causal_mask(lengths, sequence, None, tokens.device())?;
        let windowed = match self.architecture.sliding_window {
            Some(window) if self.architecture.sliding_layers != 0 => Some(padded_causal_mask(
                lengths,
                sequence,
                Some(window),
                tokens.device(),
            )?),
            _ => None,
        };
        let masks = BatchMasks { full, windowed };
        Ok(self.decode(tokens, 0, cache, steering, &[], Some(&masks), mode)?)
    }

    /// The decoder loop both entry points run.
    ///
    /// `masks`, when present, replace the cache's causal mask for every layer:
    /// a batched caller builds the combined causal and key-padding mask up
    /// front — once for full attention and once more for a sliding window, if
    /// the model has one — and every layer takes the one its attention uses,
    /// because the constraint depends only on the batch's lengths and the
    /// layer's window. `None` is the historical single-sequence path, which
    /// still asks the cache for a mask keyed by shape and window alone — a
    /// padding mask has no such key, since two batches of the same shape can
    /// pad differently, which is why it is built per call and never memoised.
    fn decode(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
        masks: Option<&BatchMasks>,
        mode: Mode,
    ) -> candle_core::Result<ForwardOutput> {
        let (_, sequence) = tokens.dims2()?;
        let mut hidden = self.embeddings.forward(tokens)?;
        if let Some(multiplier) = self.architecture.embedding_multiplier {
            // Gemma multiplies the embedding by `sqrt(hidden_size)` and Granite
            // by its `embedding_multiplier`, in the embedding's own dtype,
            // before the first block.
            hidden = (hidden * multiplier)?;
        }
        let mut activations = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let mask = masks.map(|masks| masks.for_window(layer.window()));
            hidden = layer.forward(&hidden, index_pos, index, cache, mask, mode)?;
            if capture_layers.binary_search(&index).is_ok() {
                let activation = hidden
                    .i((0, sequence - 1, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                activations.insert(index, activation);
            }
            if let Some(plan) = steering {
                if let Some(vector) = plan.vector(index) {
                    let scaled = (vector * plan.strength)?.reshape((1, 1, plan.hidden_size))?;
                    hidden = hidden.broadcast_add(&scaled).map_err(|error| {
                        error.context(format!("failed to apply steering at layer {index}"))
                    })?;
                }
            }
        }
        let hidden = normalize(&self.final_norm, &hidden, mode.pass)?;
        // Decoding only ever samples the next token, so it projects one row and
        // leaves the rest of the vocabulary matmul undone. Anything that scores
        // a sequence against its own successors needs every position, and a
        // reward head needs no vocabulary at all.
        let logits = match mode.readout {
            Readout::LastPosition => {
                let last = hidden.i((.., sequence - 1, ..))?.contiguous()?;
                Some(self.soft_cap(self.lm_head.forward(&last)?.to_dtype(DType::F32)?)?)
            }
            Readout::EveryPosition => Some(self.soft_cap(
                self.lm_head
                    .forward(&hidden.contiguous()?)?
                    .to_dtype(DType::F32)?,
            )?),
            Readout::Hidden => None,
        };
        Ok(ForwardOutput {
            logits,
            hidden: hidden.to_dtype(DType::F32)?,
            activations,
        })
    }
}

impl SteeringLlama {
    /// Granite divides its final logits by `logits_scaling`; Gemma 2 and 3
    /// bound them to `(-cap, cap)` with a tanh.
    fn soft_cap(&self, logits: Tensor) -> candle_core::Result<Tensor> {
        let logits = match self.architecture.logits_divisor {
            Some(divisor) => (logits / divisor)?,
            None => logits,
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
