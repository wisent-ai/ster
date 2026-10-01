//! The whole decoder: embeddings, the stack of blocks, the final norm and
//! the head — plus the one entry every caller runs a forward pass through.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use candle_transformers::models::llama::Config;

use crate::lora::Adapters;

use super::{
    Architecture, Cache, ForwardOutput, Mode, Readout, SteeringPlan,
    attention::padded_causal_mask,
    layer::{DecoderLayer, normalize},
};

#[derive(Debug, Clone)]
pub struct SteeringLlama {
    embeddings: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    config: Config,
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
        let final_norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
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
            adapters,
        })
    }

    pub fn adapters(&self) -> &crate::lora::Adapters {
        &self.adapters
    }

    pub fn config(&self) -> &Config {
        &self.config
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
        let mask = padded_causal_mask(lengths, sequence, tokens.device())?;
        Ok(self.decode(tokens, 0, cache, steering, &[], Some(&mask), mode)?)
    }

    /// The decoder loop both entry points run.
    ///
    /// `mask`, when present, replaces the cache's causal mask for every layer:
    /// a batched caller builds one combined causal and key-padding mask up
    /// front and hands the same handle down the stack, because the constraint
    /// depends only on the batch's lengths and so is identical at every one of
    /// the model's layers. `None` is the historical single-sequence path,
    /// which still asks the cache for a mask keyed by shape alone — a padding
    /// mask has no such key, since two batches of the same shape can pad
    /// differently, which is why it is built per call and never memoised.
    fn decode(
        &self,
        tokens: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
        steering: Option<&SteeringPlan>,
        capture_layers: &[usize],
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<ForwardOutput> {
        let (_, sequence) = tokens.dims2()?;
        let mut hidden = self.embeddings.forward(tokens)?;
        let mut activations = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
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
                Some(self.lm_head.forward(&last)?.to_dtype(DType::F32)?)
            }
            Readout::EveryPosition => Some(
                self.lm_head
                    .forward(&hidden.contiguous()?)?
                    .to_dtype(DType::F32)?,
            ),
            Readout::Hidden => None,
        };
        Ok(ForwardOutput {
            logits,
            hidden: hidden.to_dtype(DType::F32)?,
            activations,
        })
    }
}
