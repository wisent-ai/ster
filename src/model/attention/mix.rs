//! What one attention block does with its rotated query, keys and values:
//! the keys the sparse indexers hide, the attention itself (IQuest-
//! LoopCoder's gated mix of two on its later passes), and the gates on the
//! output before its projection.

use candle_core::{DType, Tensor};
use candle_nn::Module;

use super::super::{Cache, Mode};
use super::forward::shared;
use super::latent::Indexed;
use super::{Attention, gated, project};

impl Attention {
    /// The mask this call attends under once the sparse indexers have
    /// hidden keys, `None` when neither hides any and `mask` stands.
    ///
    /// DeepSeek Sparse Attention: past `index_topk` keys, each query sees
    /// only the keys its indexer ranks highest (a GLM-5 `shared` layer
    /// reuses the last indexed layer's choice in this call). MiniMax Sparse
    /// Attention: past `sparse_topk_blocks` key blocks, each query head sees
    /// only the blocks its group's index ranks highest.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sparse_mask(
        &self,
        hidden: &Tensor,
        indexed: Option<Indexed<'_>>,
        angles: Option<&(Tensor, Tensor)>,
        mask: Option<&Tensor>,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<Option<Tensor>> {
        let (_, sequence, _) = hidden.dims3()?;
        let index_hidden = match (indexed, self.index_shared) {
            (Some((indexer, latent_query)), _) => {
                let causal = match mask {
                    Some(full) => full.squeeze(1)?,
                    None => cache.mask(sequence, index_pos, self.window)?,
                };
                let hidden_keys = indexer.hidden_keys(
                    hidden,
                    &latent_query,
                    angles,
                    &causal,
                    layer,
                    cache,
                    mode,
                )?;
                cache.index_mask = hidden_keys.clone();
                hidden_keys
            }
            (None, true) => cache.index_mask.clone(),
            (None, false) => None,
        };
        let index_hidden = match &self.block_indexer {
            Some(blocks) => blocks.hidden_keys(
                hidden, angles, self.heads, index_pos, layer, cache, mode,
            )?,
            None => index_hidden,
        };
        let Some(index_hidden) = index_hidden else {
            return Ok(None);
        };
        Ok(Some(match mask {
            Some(full) => full.broadcast_maximum(&index_hidden)?,
            None => cache
                .mask(sequence, index_pos, self.window)?
                .broadcast_maximum(&index_hidden)?,
        }))
    }

    /// The attention of `query` over `key` and `value`, `[batch, heads,
    /// sequence, value_dim]`.
    ///
    /// IQuest-LoopCoder's later passes mix global attention over the first
    /// pass's keys and values with local attention over their own, within
    /// `loop_window_size`, by a per-head gate of the rotated query:
    /// `sigmoid(W_h · q_h + b_h)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mixed(
        &self,
        query: &Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let Some(mix) = self.loop_mix.as_ref().filter(|mix| layer >= mix.physical) else {
            return self.attend(query, key, value, self.window, mask, index_pos, cache, mode);
        };
        let (_, _, sequence, _) = query.dims4()?;
        let (global_key, global_value) = shared(cache, layer, layer % mix.physical)?;
        let global = self.attend(
            query,
            global_key,
            global_value,
            None,
            mask,
            index_pos,
            cache,
            mode,
        )?;
        let local_mask = match mask {
            Some(full) => Some(
                full.broadcast_maximum(
                    &cache
                        .mask(sequence, index_pos, Some(mix.window))?
                        .to_dtype(full.dtype())?,
                )?,
            ),
            None => None,
        };
        let local = self.attend(
            query,
            key,
            value,
            Some(mix.window),
            local_mask.as_ref(),
            index_pos,
            cache,
            mode,
        )?;
        let logits = query
            .to_dtype(DType::F32)?
            .broadcast_mul(&mix.weight)?
            .sum_keepdim(3)?
            .broadcast_add(&mix.bias)?;
        // sigmoid, composed so it has a backward pass.
        let gate = (logits.neg()?.exp()? + 1.0)?
            .recip()?
            .to_dtype(global.dtype())?;
        global.broadcast_mul(&gate)? + local.broadcast_mul(&(gate.neg()? + 1.0)?)?
    }

    /// `output` `[batch, heads, sequence, value_dim]` through the gates the
    /// family puts on it and the output projection, `[batch, sequence,
    /// hidden]`.
    pub(super) fn gated_output(
        &self,
        output: Tensor,
        hidden: &Tensor,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = hidden.dims3()?;
        // Step 3.5's and Laguna's head-wise gate: each head's output times
        // its `g_proj` logit through the gate function.
        let output = match &self.head_gate {
            Some((gate, function)) => {
                let logits = gate.forward(hidden)?.transpose(1, 2)?.unsqueeze(3)?;
                output.broadcast_mul(&gated(&logits, *function)?.to_dtype(output.dtype())?)?
            }
            None => output,
        };
        let output = output
            .transpose(1, 2)?
            .reshape((batch, sequence, self.heads * self.value_dim))?;
        // K2-Horizon's and AFMoE's elementwise gate.
        let output = match &self.elementwise_gate {
            Some((gate, function)) => (output * gated(&gate.forward(hidden)?, *function)?)?,
            None => output,
        };
        // sigmoid, composed so it has a backward pass.
        let output = match &self.output_gate {
            Some(gate) => {
                output.broadcast_mul(&(gate.forward(hidden)?.neg()?.exp()? + 1.0)?.recip()?)?
            }
            None => output,
        };
        project(&self.output, self.output_adapter.as_ref(), &output, mode.route)
    }
}
