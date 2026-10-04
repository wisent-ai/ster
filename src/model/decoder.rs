//! The whole decoder: embeddings, the stack of blocks, the final norm and
//! the head — plus the one entry every caller runs a forward pass through.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding};
use candle_transformers::models::llama::Config;

use crate::lora::Adapters;

use super::{
    Architecture, Cache, ForwardOutput, Mode, Positions, Readout, SteeringPlan,
    attention::padded_causal_mask,
    layer::{
        DecoderLayer, LayerInputs,
        norm::{Norm, NormSpec},
        projection,
        shared::SharedBlock,
    },
};

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
    final_norm: Norm,
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
        mut architecture: Architecture,
        adapters: crate::lora::Adapters,
    ) -> candle_core::Result<Self> {
        // A multimodal checkpoint keeps the language model below a wrapper
        // (Gemma 3's `language_model`); the vision tower beside it is never
        // mapped.
        let builder = if architecture.names.wrapper.is_empty() {
            builder
        } else {
            builder.pp(architecture.names.wrapper)
        };
        // A checkpoint saved from the base model drops the family's root
        // (`wte` rather than `transformer.wte`); which one this is shows in
        // whether the embedding is where the root says.
        if !builder.contains_tensor(&format!("{}.weight", architecture.names.embeddings)) {
            architecture.names = architecture.names.without_root();
        }
        let names = architecture.names;
        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            builder.pp(names.embeddings),
        )?;
        let positions = match architecture.positions {
            Positions::Learned { offset } => Some((
                embedding(
                    config.max_position_embeddings + offset,
                    config.hidden_size,
                    builder.pp(names.positions),
                )?,
                offset,
            )),
            _ => None,
        };
        let spec = NormSpec::of(&architecture);
        let embedding_norm = if architecture.embedding_norm {
            Some(spec.load(config.hidden_size, builder.pp(names.embedding_norm))?)
        } else {
            None
        };
        let lm_head = if config.tie_word_embeddings {
            let bias = if architecture.lm_head_bias {
                Some(builder.pp(names.lm_head).get(config.vocab_size, "bias")?)
            } else {
                None
            };
            Linear::new(embeddings.embeddings().clone(), bias)
        } else {
            projection(
                config.hidden_size,
                config.vocab_size,
                architecture.lm_head_bias,
                false,
                builder.pp(names.lm_head),
            )?
        };
        let final_norm = spec.load(config.hidden_size, builder.pp(names.final_norm))?;
        let per_layer = match architecture.per_layer_input {
            Some(per_layer) => {
                let root = if names.root.is_empty() { builder.clone() } else { builder.pp(names.root) };
                let packed = config.num_hidden_layers * per_layer.width;
                Some(PerLayerEmbeddings {
                    embeddings: embedding(per_layer.vocab, packed, root.pp("embed_tokens_per_layer"))?,
                    projection: candle_nn::linear_no_bias(
                        config.hidden_size,
                        packed,
                        root.pp("per_layer_model_projection"),
                    )?,
                    norm: spec.load(per_layer.width, root.pp("per_layer_projection_norm"))?,
                    width: per_layer.width,
                })
            }
            None => None,
        };
        // Zamba2's shared blocks are mapped once, from the first hybrid
        // layers that use them, and every later use shares their tensors.
        let shared = match architecture.shared_blocks {
            Some(blocks) => {
                let owners: Vec<usize> = blocks.layers().take(blocks.blocks).collect();
                owners
                    .iter()
                    .map(|owner| {
                        let block_builder = builder.pp(format!("{}.{owner}.shared_transformer", names.layers));
                        let block = SharedBlock::load(&block_builder, &config, &architecture, *owner, blocks)?;
                        Ok((block, block_builder))
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?
            }
            None => Vec::new(),
        };
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                let block = architecture
                    .shared_blocks
                    .filter(|blocks| index < u128::BITS as usize && blocks.hybrid_layers & (1u128 << index) != 0)
                    .and_then(|blocks| shared.get(blocks.slot(index) % blocks.blocks))
                    .map(|(block, block_builder)| (block, block_builder));
                DecoderLayer::load(
                    builder.pp(format!("{}.{index}", names.layers)),
                    &config,
                    &architecture,
                    index,
                    &adapters,
                    block,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            embeddings,
            positions,
            embedding_norm,
            layers,
            per_layer,
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
        if let Some((table, offset)) = &self.positions {
            let first = (index_pos + offset) as u32;
            let rows = Tensor::arange(first, first + sequence as u32, tokens.device())?;
            hidden = hidden.broadcast_add(&table.forward(&rows)?.unsqueeze(0)?)?;
        }
        if let Some(norm) = &self.embedding_norm {
            hidden = norm.forward(&hidden, mode.pass)?;
        }
        if let Some(multiplier) = self.architecture.embedding_multiplier {
            // Gemma multiplies the embedding by `sqrt(hidden_size)` and Granite
            // by its `embedding_multiplier`, in the embedding's own dtype,
            // before the first block.
            hidden = (hidden * multiplier)?;
        }
        // Zamba2's hybrid layers read the embeddings beside the hidden state;
        // Gemma 4 projects its per-layer inputs from them.
        let embedded = hidden.clone();
        let per_layer = match &self.per_layer {
            Some(per_layer) => Some(per_layer.inputs(tokens, &embedded, self.layers.len(), mode)?),
            None => None,
        };
        let mut activations = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let mask = masks.map(|masks| masks.for_window(layer.window()));
            let inputs = LayerInputs {
                embedded: &embedded,
                per_layer: per_layer.as_ref().map(|all| all.i((.., .., index, ..))).transpose()?,
            };
            hidden = layer.forward(&hidden, &inputs, index_pos, index, cache, mask, mode)?;
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
        let hidden = self.final_norm.forward(&hidden, mode.pass)?;
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
    /// Cohere multiplies its final logits by `logit_scale` and Granite divides
    /// them by `logits_scaling`; Gemma 2 and 3 bound them to `(-cap, cap)`
    /// with a tanh.
    fn soft_cap(&self, logits: Tensor) -> candle_core::Result<Tensor> {
        let logits = match self.architecture.logits_multiplier {
            Some(multiplier) => (logits * multiplier)?,
            None => logits,
        };
        match self.architecture.final_softcap {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }
    }
}

/// Gemma 4's per-layer input tables, below the model root:
/// `embed_tokens_per_layer`, `per_layer_model_projection` and
/// `per_layer_projection_norm`.
#[derive(Debug, Clone)]
struct PerLayerEmbeddings {
    embeddings: Embedding,
    projection: Linear,
    norm: Norm,
    width: usize,
}

impl PerLayerEmbeddings {
    /// Every layer's input, `[batch, sequence, layers, width]`: the token's
    /// row of `embed_tokens_per_layer` times `sqrt(width)`, plus the scaled
    /// embeddings projected, times `hidden_size^-0.5` and normed, the sum
    /// times `2^-0.5`, as Transformers' `get_per_layer_inputs` and
    /// `project_per_layer_inputs` compute them.
    fn inputs(&self, tokens: &Tensor, embedded: &Tensor, layers: usize, mode: Mode) -> candle_core::Result<Tensor> {
        let (batch, sequence) = tokens.dims2()?;
        let hidden = embedded.dim(candle_core::D::Minus1)?;
        let shape = (batch, sequence, layers, self.width);
        let identity = (self.embeddings.forward(tokens)? * (self.width as f64).sqrt())?.reshape(shape)?;
        let context = (self.projection.forward(embedded)? * (hidden as f64).powf(-0.5))?.reshape(shape)?;
        let context = self.norm.forward(&context, mode.pass)?;
        (context + identity)? * std::f64::consts::FRAC_1_SQRT_2
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
