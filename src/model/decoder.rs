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
        altup::StreamProjections,
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
    /// LongCat-Flash-Lite's n-gram tables.
    ngram: Option<NgramEmbeddings>,
    /// Gemma 3n's projections into and out of its AltUp streams.
    streams: Option<StreamProjections>,
    /// HRM-Text's starting low state (`model.z_L_init`).
    low_start: Option<Tensor>,
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
        // The head may sit outside it, at the checkpoint's root (Qwen3.5's
        // and MuseGlimmer's `lm_head`).
        let outer = builder.clone();
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
        // BLOOM's embedding norm is stored; MuseGlimmer's is weightless and
        // names no tensor.
        let embedding_norm = if architecture.embedding_norm {
            Some(if names.embedding_norm.is_empty() {
                spec.unscaled(config.hidden_size, &builder)?
            } else {
                spec.load(config.hidden_size, builder.pp(names.embedding_norm))?
            })
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
            let head = if builder.contains_tensor(&format!("{}.weight", names.lm_head)) { &builder } else { &outer };
            projection(
                config.hidden_size,
                config.vocab_size,
                architecture.lm_head_bias,
                false,
                head.pp(names.lm_head),
            )?
        };
        let final_spec = if architecture.plain_final_norm { NormSpec { offset: false, ..spec } } else { spec };
        let final_norm = final_spec.load(config.hidden_size, builder.pp(names.final_norm))?;
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
        let ngram = match architecture.ngram {
            Some(spec) => {
                let root = if names.root.is_empty() { builder.clone() } else { builder.pp(names.root) };
                Some(NgramEmbeddings::load(spec, config.vocab_size, config.hidden_size, root.pp("ngram_embeddings"))?)
            }
            None => None,
        };
        let streams = match architecture.altup_streams {
            Some(count) => {
                let root = if names.root.is_empty() { builder.clone() } else { builder.pp(names.root) };
                Some(StreamProjections::load(&root, config.hidden_size, count)?)
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
        // A looped model (Nanbeige's `num_loops`) runs its stored layers
        // more than once; each later pass reuses the first pass's tensors.
        // LongCat-Flash's stored layers are two Ster layers each, read under
        // their halves' names.
        // HRM-Text's two stacks are stored apart, each below its own path.
        let stored = match (architecture.recurrence, architecture.loops) {
            (Some(recurrence), _) => 2 * recurrence.per_stack,
            (None, Some(loops)) => loops.physical,
            (None, None) => config.num_hidden_layers,
        };
        let physical = (0..stored)
            .map(|index| {
                let block = architecture
                    .shared_blocks
                    .filter(|blocks| index < u128::BITS as usize && blocks.hybrid_layers & (1u128 << index) != 0)
                    .and_then(|blocks| shared.get(blocks.slot(index) % blocks.blocks))
                    .map(|(block, block_builder)| (block, block_builder));
                let (source, layer_names) = architecture.stored_layer(index);
                let half;
                let layer_architecture = if layer_names == architecture.names {
                    &architecture
                } else {
                    half = Architecture { names: layer_names, ..architecture.clone() };
                    &half
                };
                DecoderLayer::load(
                    builder.pp(format!(
                        "{}.{source}",
                        if architecture.recurrence.is_some() { layer_names.layers } else { names.layers }
                    )),
                    &config,
                    layer_architecture,
                    index,
                    &adapters,
                    block,
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| match architecture.recurrence {
                Some(recurrence) => physical[recurrence.stored(index)].clone(),
                None => physical[index % stored].clone(),
            })
            .collect();
        let low_start = match architecture.recurrence {
            Some(_) => Some(builder.get(config.hidden_size, "model.z_L_init")?),
            None => None,
        };
        Ok(Self {
            embeddings,
            positions,
            embedding_norm,
            layers,
            per_layer,
            ngram,
            streams,
            low_start,
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
        if let Some(ngram) = &self.ngram {
            hidden = ngram.embed(tokens, hidden, index_pos, cache)?;
        }
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
        // Solar's block skip connections keep up to two earlier hidden states.
        let mut kept: [Option<Tensor>; 2] = [None, None];
        // Gemma 3n's AltUp streams beside the first, which is `hidden`.
        let mut rest = match &self.streams {
            Some(streams) => Some(streams.spread(&hidden)?),
            None => None,
        };
        // HRM-Text's high and low states: the high one starts as the scaled
        // embedding, the low one as `z_L_init` at every position.
        let mut states = match (self.architecture.recurrence, &self.low_start) {
            (Some(_), Some(low)) => {
                Some((hidden.clone(), low.to_dtype(hidden.dtype())?.broadcast_as(hidden.shape())?.contiguous()?))
            }
            _ => None,
        };
        for (index, layer) in self.layers.iter().enumerate() {
            // A pass of either stack reads the two states summed.
            if let (Some(recurrence), Some((high, low))) = (self.architecture.recurrence, &states) {
                if index % recurrence.per_stack == 0 {
                    hidden = (high + low)?;
                }
            }
            // Nanbeige normalises the hidden state by the final norm after
            // every pass but the last, whose norm is the model's own.
            if let Some(loops) = self.architecture.loops {
                if loops.norm_between && index > 0 && index % loops.physical == 0 {
                    hidden = self.final_norm.forward(&hidden, mode.pass)?;
                }
            }
            if let Some(skips) = &self.architecture.skip_connections {
                hidden = skips.apply(index, hidden, &mut kept)?;
            }
            let mask = masks.map(|masks| masks.for_window(layer.window()));
            let inputs = LayerInputs {
                embedded: &embedded,
                per_layer: per_layer.as_ref().map(|all| all.i((.., .., index, ..))).transpose()?,
            };
            hidden = match rest.take() {
                Some(others) => {
                    let mut all = Vec::with_capacity(others.len() + 1);
                    all.push(hidden);
                    all.extend(others);
                    let mut corrected = layer.forward_streams(&all, &inputs, index_pos, index, cache, mask, mode)?;
                    let first = corrected.remove(0);
                    rest = Some(corrected);
                    first
                }
                None => layer.forward(&hidden, &inputs, index_pos, index, cache, mask, mode)?,
            };
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
            // The pass ends in the stack's scale-free norm and replaces its
            // own state.
            if let (Some(recurrence), Some((high, low))) = (self.architecture.recurrence, states.as_mut()) {
                if index % recurrence.per_stack == recurrence.per_stack - 1 {
                    hidden = self.final_norm.forward(&hidden, mode.pass)?;
                    if recurrence.low(index) {
                        *low = hidden.clone();
                    } else {
                        *high = hidden.clone();
                    }
                }
            }
        }
        let hidden = match (&self.streams, &rest) {
            (Some(streams), Some(others)) => streams.join(&hidden, others)?,
            _ => hidden,
        };
        // HRM-Text's last pass already ended in its norm.
        let hidden = if states.is_some() { hidden } else { self.final_norm.forward(&hidden, mode.pass)? };
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

/// LongCat's n-gram tables, below the model root: `embedders.{i}` and
/// `post_projs.{i}` for each of `splits · (neighbors - 1)` tables, as
/// LongCat-Flash-Lite's `modeling_longcat_ngram.py` names them.
#[derive(Debug, Clone)]
struct NgramEmbeddings {
    tables: Vec<(Embedding, Linear)>,
    spec: super::NgramSpec,
    vocab: u64,
}

impl NgramEmbeddings {
    fn load(spec: super::NgramSpec, vocab: usize, hidden: usize, builder: VarBuilder<'_>) -> candle_core::Result<Self> {
        let count = spec.splits * spec.neighbors.saturating_sub(1);
        if count == 0 || hidden % count != 0 {
            candle_core::bail!(
                "{count} n-gram tables cannot split a hidden width of {hidden} evenly (emb_split_num times emb_neighbor_num - 1)"
            );
        }
        let width = hidden / count;
        let tables = (0..count)
            .map(|index| -> candle_core::Result<(Embedding, Linear)> {
                Ok((
                    embedding(Self::rows(spec, vocab, index), width, builder.pp(format!("embedders.{index}")))?,
                    candle_nn::linear_no_bias(width, hidden, builder.pp(format!("post_projs.{index}")))?,
                ))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self { tables, spec, vocab: vocab as u64 })
    }

    /// Table `index`'s height, which is also the modulus its hashes reduce by.
    fn rows(spec: super::NgramSpec, vocab: usize, index: usize) -> usize {
        spec.ratio * vocab + 2 * index + 1
    }

    /// `embedded` (the token embeddings of `tokens`) averaged with every
    /// table's projected row, the n-grams reaching back into the tokens the
    /// cache kept from the previous call when this call continues one.
    /// Hashing follows `NgramEmbedding.forward`: for order `i` the id is
    /// `token + Σ_{d=1}^{i-1} token_{-d} · vocab^d mod rows`, reduced mod
    /// `rows`, a token before the start of its run (the sequence, or the
    /// last `eos` up to and including it) counting as zero.
    fn embed(&self, tokens: &Tensor, embedded: Tensor, index_pos: usize, cache: &mut super::Cache) -> candle_core::Result<Tensor> {
        let (batch, sequence) = tokens.dims2()?;
        let rows: Vec<Vec<u32>> = tokens.to_dtype(DType::U32)?.to_vec2()?;
        let kept = self.spec.neighbors - 1;
        let earlier = cache.ngram_context.take().filter(|_| cache.use_kv_cache && index_pos > 0);
        let contexts: Vec<Vec<u32>> = rows
            .iter()
            .enumerate()
            .map(|(row, tokens)| {
                let mut context = earlier.as_ref().and_then(|earlier| earlier.get(row)).cloned().unwrap_or_default();
                context.extend_from_slice(tokens);
                context
            })
            .collect();
        if cache.use_kv_cache {
            cache.ngram_context = Some(
                contexts.iter().map(|context| context[context.len().saturating_sub(kept)..].to_vec()).collect(),
            );
        }
        let device = tokens.device();
        let mut sum = embedded.clone();
        for order in 2..=self.spec.neighbors {
            for split in 0..self.spec.splits {
                let index = (order - 2) * self.spec.splits + split;
                let modulus = Self::rows(self.spec, self.vocab as usize, index) as u64;
                let mut ids = Vec::with_capacity(batch * sequence);
                for context in &contexts {
                    let start = context.len() - sequence;
                    let mut run_start = 0;
                    for (position, token) in context.iter().enumerate() {
                        if position >= start {
                            let mut id = u64::from(*token);
                            let mut power = 1u64;
                            for back in 1..order {
                                power = power * self.vocab % modulus;
                                if position >= run_start + back {
                                    id += u64::from(context[position - back]) * power;
                                }
                            }
                            ids.push((id % modulus) as u32);
                        }
                        if *token == self.spec.eos {
                            run_start = position + 1;
                        }
                    }
                }
                let ids = Tensor::from_vec(ids, (batch, sequence), device)?;
                let (table, projection) = &self.tables[index];
                sum = (sum + projection.forward(&table.forward(&ids)?)?.to_dtype(embedded.dtype())?)?;
            }
        }
        sum / (1 + self.tables.len()) as f64
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
