//! The decoder loop every forward pass runs: the embeddings, every layer in
//! the residual form its family keeps, steering, and the readout.

use std::collections::BTreeMap;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::Module;

use super::super::{Cache, ForwardOutput, Mode, Readout, SteeringPlan, layer::LayerInputs};
use super::{BatchMasks, SteeringLlama};

impl SteeringLlama {
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
    pub(super) fn decode(
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
            Some(per_layer) => {
                Some(per_layer.inputs(tokens, &embedded, self.layers.len(), mode)?)
            }
            None => None,
        };
        let mut activations = BTreeMap::new();
        let mut positions = BTreeMap::new();
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
            (Some(_), Some(low)) => Some((
                hidden.clone(),
                low.to_dtype(hidden.dtype())?
                    .broadcast_as(hidden.shape())?
                    .contiguous()?,
            )),
            _ => None,
        };
        // Kimi-K3's finished block sums; `hidden` is the running sum.
        let mut blocks: Vec<Tensor> = Vec::new();
        // DeepSeek-V4's and GLM-5-Next's residual streams, each the
        // embedding at first; `hidden` is their mean, and steering moves
        // every one of them.
        let mut hyper = match self.architecture.hyper_connections {
            Some(spec) => {
                let (batch, sequence, width) = hidden.dims3()?;
                Some(
                    hidden
                        .unsqueeze(2)?
                        .expand((batch, sequence, spec.streams, width))?
                        .contiguous()?,
                )
            }
            _ => None,
        };
        if hyper.is_some()
            || self
                .architecture
                .experts
                .as_ref()
                .is_some_and(|experts| experts.hash_layers != 0)
        {
            cache.token_ids = Some(tokens.flatten_all()?.to_vec1::<u32>()?);
        }
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
                per_layer: per_layer
                    .as_ref()
                    .map(|all| all.i((.., .., index, ..)))
                    .transpose()?,
            };
            hidden = match (rest.take(), hyper.take()) {
                (Some(others), held) => {
                    hyper = held;
                    let mut all = Vec::with_capacity(others.len() + 1);
                    all.push(hidden);
                    all.extend(others);
                    let mut corrected = layer
                        .forward_streams(&all, &inputs, index_pos, index, cache, mask, mode)?;
                    let first = corrected.remove(0);
                    rest = Some(corrected);
                    first
                }
                (None, Some(held)) => {
                    let streams = layer.forward_hyper(
                        &held, &inputs, index_pos, index, cache, mask, mode,
                    )?;
                    let mean = streams.mean(2)?;
                    hyper = Some(streams);
                    mean
                }
                (None, None) if self.depth_output.is_some() => layer.forward_depth(
                    &hidden,
                    &mut blocks,
                    &inputs,
                    index_pos,
                    index,
                    cache,
                    mask,
                    mode,
                )?,
                (None, None) => {
                    layer.forward(&hidden, &inputs, index_pos, index, cache, mask, mode)?
                }
            };
            if capture_layers.binary_search(&index).is_ok() {
                let activation = hidden
                    .i((0, sequence - 1, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                activations.insert(index, activation);
                // Every position's state, `[sequence, hidden]`: a view of the
                // tensor the layer wrote, so no copy is made until it is read.
                positions.insert(index, hidden.i(0)?);
            }
            if let Some(plan) = steering {
                if let Some(vector) = plan.vector(index) {
                    let scaled = (vector * plan.strength)?.reshape((1, 1, plan.hidden_size))?;
                    hidden = hidden.broadcast_add(&scaled)?;
                    if let Some(streams) = hyper.take() {
                        hyper = Some(streams.broadcast_add(&scaled.unsqueeze(2)?)?);
                    }
                }
            }
            // The pass ends in the stack's scale-free norm and replaces its
            // own state.
            if let (Some(recurrence), Some((high, low))) =
                (self.architecture.recurrence, states.as_mut())
            {
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
        let hidden = match &self.depth_output {
            Some(mix) => mix.mix(&blocks, &hidden)?,
            None => hidden,
        };
        let hidden = match (&self.hyper_head, &hyper) {
            (Some(head), Some(streams)) => head.collapse(streams)?,
            (None, Some(streams)) => streams.mean(2)?,
            _ => hidden,
        };
        // HRM-Text's last pass already ended in its norm.
        let hidden = if states.is_some() {
            hidden
        } else {
            self.final_norm.forward(&hidden, mode.pass)?
        };
        // Decoding only ever samples the next token, so it projects one row and
        // leaves the rest of the vocabulary matmul undone. Anything that scores
        // a sequence against its own successors needs every position, and a
        // reward head needs no vocabulary at all.
        let logits = match mode.readout {
            Readout::LastPosition => {
                let last = hidden.i((.., sequence - 1, ..))?.contiguous()?;
                Some(self.soft_cap(self.lm_head.forward(&last)?.to_dtype(DType::F32)?)?)
            }
            Readout::EveryPosition => Some(
                self.soft_cap(
                    self.lm_head
                        .forward(&hidden.contiguous()?)?
                        .to_dtype(DType::F32)?,
                )?,
            ),
            Readout::Hidden => None,
        };
        Ok(ForwardOutput {
            logits,
            hidden: hidden.to_dtype(DType::F32)?,
            activations,
            positions,
        })
    }
}
