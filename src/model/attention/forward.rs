//! One attention block's forward pass: its query, key and value, their
//! rotation and cache, the sparse indexers' masks, and the gates on its
//! output.

use candle_core::{DType, Tensor};
use candle_nn::Module;

use crate::failures::{model::decoder::KEYS_UNSHARED, refuse_candle};

use super::super::{Cache, Mode, QueryKeyNorm, QueryTemperature};
use super::latent::Indexed;
use super::{Attention, Projections, Rotary, optional_norm, project, rope::apply_rotary};

/// Where one call's keys and values come from: this layer's own
/// projections, `[batch, sequence, key_value_heads, width]` before the
/// rotation, or the layer a Gemma 4 key-value-sharing layer reuses.
enum KeyValue {
    Own(Tensor, Tensor),
    Shared(usize),
}

/// A rotary table's angles for exactly the positions one call rotates.
type Angles = (Tensor, Tensor);

impl Attention {
    /// One attention block, optionally under a caller-supplied mask.
    ///
    /// `mask` is the combined causal and key-padding constraint a batched
    /// caller built once for the whole stack, shaped `[batch, 1, sequence,
    /// keys]` so the unit head axis broadcasts across every head. When it is
    /// `None` this is the single-sequence path and the mask comes from the
    /// cache, exactly as it always did.
    pub(in crate::model) fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut Cache,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (_, sequence, _) = hidden.dims3()?;
        let (query, key_value, indexed) = self.query_key_value(hidden, mode)?;
        let angles = self.angles(cache, index_pos, sequence)?;
        let query = self.rotated_query(query, angles.as_ref(), index_pos, cache.weights, mode)?;
        let (key, value) = self.keys_values(key_value, angles.as_ref(), layer, cache, mode)?;
        let sparse = self.sparse_mask(
            hidden,
            indexed,
            angles.as_ref(),
            mask,
            index_pos,
            layer,
            cache,
            mode,
        )?;
        let mask = sparse.as_ref().or(mask);
        let output = self.mixed(&query, key, value, mask, index_pos, layer, cache, mode)?;
        self.gated_output(output, hidden, mode)
    }

    /// The query, `[batch, sequence, heads, head_dim]`, where this call's
    /// keys and values come from, and the sparse indexer with the latent
    /// query it reads.
    fn query_key_value<'a>(
        &'a self,
        hidden: &Tensor,
        mode: Mode,
    ) -> candle_core::Result<(Tensor, KeyValue, Option<Indexed<'a>>)> {
        let (batch, sequence, _) = hidden.dims3()?;
        let (query, key, value) = match &self.projections {
            Projections::Standard { query, key, value } => {
                let query = match &self.query_bottleneck {
                    Some((down, norm)) => {
                        let narrowed = norm.forward(&down.forward(hidden)?, mode.pass)?;
                        project(query, self.query_adapter.as_ref(), &narrowed, mode.route)?
                    }
                    None => project(query, self.query_adapter.as_ref(), hidden, mode.route)?,
                };
                let key = project(key, self.key_adapter.as_ref(), hidden, mode.route)?;
                // Gemma 4's full-attention layers under `attention_k_eq_v`
                // take the value from the key projection, before any norm;
                // K2-Horizon's routed layers from their value experts.
                let value = match (value, &self.value_experts) {
                    (Some(value), _) => {
                        project(value, self.value_adapter.as_ref(), hidden, mode.route)?
                    }
                    (None, Some(experts)) => experts.forward(hidden)?,
                    (None, None) => key.clone(),
                };
                (query, key, value)
            }
            Projections::Latent(latent) => {
                let (query, key, value, indexed) =
                    latent.forward(hidden, self.heads, self.query_adapter.as_ref(), mode)?;
                return Ok((query, KeyValue::Own(key, value), indexed));
            }
            Projections::QueryOnly { query, source } => {
                let query = project(query, self.query_adapter.as_ref(), hidden, mode.route)?
                    .reshape((batch, sequence, self.heads, self.head_dim))?;
                return Ok((query, KeyValue::Shared(*source), None));
            }
        };
        // Zamba2's per-invocation low-rank terms belong to the base model,
        // so they apply on every route.
        let (query, key, value) = match &self.low_rank {
            Some(low_rank) => (
                (query + low_rank[0].forward(hidden)?)?,
                (key + low_rank[1].forward(hidden)?)?,
                (value + low_rank[2].forward(hidden)?)?,
            ),
            None => (query, key, value),
        };
        // Falcon-H1 multiplies every key by `key_multiplier`, MiMo-V2 every
        // value by `attention_value_scale`.
        let key = match self.key_scale {
            Some(scale) => (key * scale)?,
            None => key,
        };
        let value = match self.value_scale {
            Some(scale) => (value * scale)?,
            None => value,
        };
        // OLMo 2 normalises the whole projection before it is split into
        // heads; every other query and key norm works per head after it.
        let (query, key) = if self.query_key_norm == QueryKeyNorm::Full {
            (
                optional_norm(self.query_norm.as_ref(), query, mode.pass)?,
                optional_norm(self.key_norm.as_ref(), key, mode.pass)?,
            )
        } else {
            (query, key)
        };
        // OLMo, OLMoE and DBRX clip every query, key and value component to
        // `±clip_qkv` (after OLMoE's norms).
        let clip = |projected: Tensor| match self.clip_qkv {
            Some(limit) => projected.clamp(-limit, limit),
            None => Ok(projected),
        };
        let heads = (batch, sequence, self.key_value_heads, self.head_dim);
        let values = (batch, sequence, self.key_value_heads, self.value_dim);
        Ok((
            clip(query)?.reshape((batch, sequence, self.heads, self.head_dim))?,
            KeyValue::Own(clip(key)?.reshape(heads)?, clip(value)?.reshape(values)?),
            None,
        ))
    }

    /// The angles this layer rotates the call's positions by, `None` on a
    /// layer that does not rotate. LongRoPE switches every position to the
    /// long factors once the sequence runs past the original context, as
    /// Phi-3 does.
    fn angles(
        &self,
        cache: &Cache,
        index_pos: usize,
        sequence: usize,
    ) -> candle_core::Result<Option<Angles>> {
        let table = match self.rotary {
            Rotary::Global => match &cache.long {
                Some((long, original)) if index_pos + sequence > *original => Some(long),
                _ => Some(&cache.global),
            },
            Rotary::Local => cache.local.as_ref(),
            Rotary::None => None,
        };
        table
            .map(|table| table.angles(index_pos, sequence))
            .transpose()
    }

    /// The query `[batch, heads, sequence, head_dim]`: normed, rotated and
    /// tempered as the family does, at the weights' dtype.
    fn rotated_query(
        &self,
        query: Tensor,
        angles: Option<&Angles>,
        index_pos: usize,
        weights: DType,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (_, sequence, _, _) = query.dims4()?;
        let query = if self.norm_before_rotary() {
            optional_norm(self.query_norm.as_ref(), query, mode.pass)?
        } else {
            query
        };
        let query = query.transpose(1, 2)?.contiguous()?;
        let query = match angles {
            Some((cos, sin)) => {
                apply_rotary(&query, cos, sin, weights, mode.pass, self.interleaved)?
            }
            None => query,
        };
        let query = if self.norm_after_rotary {
            optional_norm(self.query_norm.as_ref(), query, mode.pass)?
        } else {
            query
        };
        // Llama 4's query temperature: each query at position `p` times
        // `1 + beta · ln(1 + floor((p + shift) / interval))`.
        let Some(QueryTemperature {
            beta,
            interval,
            shift,
            ..
        }) = self.query_temperature
        else {
            return Ok(query);
        };
        let scales: Vec<f32> = (index_pos..index_pos + sequence)
            .map(|position| {
                (1.0 + beta * (1.0 + ((position + shift) / interval.max(1)) as f64).ln()) as f32
            })
            .collect();
        let scales = Tensor::from_vec(scales, (1, 1, sequence, 1), query.device())?
            .to_dtype(query.dtype())?;
        query.broadcast_mul(&scales)
    }

    /// Whether the per-head query and key norms run before the rotation.
    fn norm_before_rotary(&self) -> bool {
        self.query_key_norm != QueryKeyNorm::Full && !self.norm_after_rotary
    }

    /// The keys and values this call attends over, `[batch, key_value_heads,
    /// keys, width]`, history included: this layer's own, rotated, cached and
    /// left for the layers that reuse them, or the ones the source layer of
    /// a key-value-sharing layer left earlier in this same call.
    fn keys_values(
        &self,
        key_value: KeyValue,
        angles: Option<&Angles>,
        layer: usize,
        cache: &mut Cache,
        mode: Mode,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (key, value) = match key_value {
            KeyValue::Own(key, value) => (key, value),
            KeyValue::Shared(source) => return shared(cache, layer, source),
        };
        let key = if self.norm_before_rotary() {
            optional_norm(self.key_norm.as_ref(), key, mode.pass)?
        } else {
            key
        };
        let value = optional_norm(self.value_norm.as_ref(), value, mode.pass)?;
        let mut key = key.transpose(1, 2)?.contiguous()?;
        let mut value = value.transpose(1, 2)?.contiguous()?;
        if let Some((cos, sin)) = angles {
            key = apply_rotary(&key, cos, sin, cache.weights, mode.pass, self.interleaved)?;
        }
        if self.norm_after_rotary {
            key = optional_norm(self.key_norm.as_ref(), key, mode.pass)?;
        }
        if cache.use_kv_cache {
            if let Some((cached_key, cached_value)) = &cache.kvs[layer] {
                key = Tensor::cat(&[cached_key, &key], 2)?.contiguous()?;
                value = Tensor::cat(&[cached_value, &value], 2)?.contiguous()?;
            }
            cache.kvs[layer] = Some((key.clone(), value.clone()));
        }
        // Gemma 4's source layers, and IQuest-LoopCoder's first pass, leave
        // every key and value they saw for the layers that reuse them.
        let first_pass = self.loop_mix.as_ref().is_some_and(|mix| layer < mix.physical);
        if self.stores_key_values || first_pass {
            cache.shared[layer] = Some((key.clone(), value.clone()));
        }
        Ok((key, value))
    }
}

/// The keys and values `source` left in the cache earlier in this call for
/// `layer` to reuse. The decoder runs its layers in order and every source
/// layer stores what it saw, so a slot found empty is refused as declared
/// (`keys-unshared`) rather than attended over as nothing.
pub(super) fn shared(
    cache: &Cache,
    layer: usize,
    source: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    cache.shared[source].clone().ok_or_else(|| {
        refuse_candle(
            &KEYS_UNSHARED,
            &[
                ("layer", &layer.to_string()),
                ("source", &source.to_string()),
            ],
        )
    })
}
