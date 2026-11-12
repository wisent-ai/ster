//! T5's learned scalar position bias, shared by all blocks in one stack.
//! The encoder distinguishes both directions; the decoder only counts past keys.

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::failures::model::{
    Unrunnable, settings,
    t5::{overflow, unreserved},
};
use crate::failures::tensor::computing;

#[derive(Debug, Clone)]
pub(in crate::model::t5) struct RelativeBias {
    weight: Tensor,
    heads: usize,
    bidirectional: bool,
    direction_width: u32,
    distances: Vec<u32>,
    saturated: u32,
}

impl RelativeBias {
    pub fn new(weight: Tensor, max_distance: usize, bidirectional: bool) -> Result<Self> {
        let (buckets, heads) = weight
            .dims2()
            .map_err(|error| computing("reading the T5 relative attention bias's axes", error))?;
        // Signed encoder distances share the available buckets equally.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        let per_direction = if bidirectional { buckets / 2 } else { buckets };
        // Within each direction, half are exact and half logarithmic.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        let exact = per_direction / 2;
        // The logarithmic partition divides by exact and by log(max_distance / exact).
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        if exact == 0 || max_distance <= exact {
            return Err(settings(
                Unrunnable::Contradictory,
                format!(
                    "T5 relative attention cannot partition {buckets} buckets at maximum distance {max_distance}: the exact-distance range of {exact} must be nonempty and shorter than the maximum distance"
                ),
                format!(
                    "relative_attention_num_buckets: {buckets}, relative_attention_max_distance: {max_distance}"
                ),
            ));
        }
        let index = |count: usize| {
            u32::try_from(count).map_err(|_| {
                overflow(
                    "the relative attention buckets",
                    &format!("{count} buckets; a bucket index is a u32"),
                )
            })
        };
        let direction_width = index(per_direction)?;
        // Distances at and beyond the maximum use the final bucket in their direction.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        let saturated = index(per_direction - 1)?;
        // The largest table index must remain representable by Candle's index tensor.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        index(buckets - 1)?;
        let denominator = ((max_distance as f64) / (exact as f64)).ln() as f32;
        let mut distances = Vec::new();
        distances.try_reserve_exact(max_distance).map_err(|error| {
            unreserved("the relative-distance lookup", max_distance, &error)
        })?;
        // The reference evaluates distance arithmetic in F32, then truncates.
        // Compute each distance once at load, not once per attention matrix cell.
        // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
        for distance in 0..max_distance {
            let bucket = if distance < exact {
                distance
            } else {
                let fraction = ((distance as f32) / (exact as f32)).ln() / denominator;
                (exact + (fraction * ((per_direction - exact) as f32)) as usize)
                    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
                    .min(per_direction - 1)
            };
            distances.push(index(bucket)?);
        }
        Ok(Self {
            weight: weight.to_dtype(DType::F32)?,
            heads,
            bidirectional,
            direction_width,
            distances,
            saturated,
        })
    }

    /// `[1, heads, queries, keys]`; `start` is the absolute decoder query position.
    pub fn forward(&self, queries: usize, keys: usize, start: usize) -> Result<Tensor> {
        let Some(end) = start.checked_add(queries) else {
            return Err(overflow(
                "the position bias",
                &format!("position {start} plus {queries} queries"),
            ));
        };
        let Some(cells) = queries.checked_mul(keys) else {
            return Err(overflow(
                "the position bias",
                &format!("{queries} queries by {keys} keys"),
            ));
        };
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(cells)
            .map_err(|error| unreserved("the position-bias indices", cells, &error))?;
        for query in start..end {
            // Memory positions count from the beginning of the cached key sequence.
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            for key in 0..keys {
                let distance = if self.bidirectional {
                    key.abs_diff(query)
                } else {
                    query.saturating_sub(key)
                };
                let mut bucket = if distance < self.distances.len() {
                    self.distances[distance]
                } else {
                    self.saturated
                };
                if self.bidirectional && key > query {
                    bucket += self.direction_width;
                }
                indices.push(bucket);
            }
        }
        let indices = Tensor::from_vec(indices, cells, self.weight.device())?;
        Ok(self
            .weight
            // The learned table has bucket rows and head columns.
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            .index_select(&indices, 0)?
            .reshape((queries, keys, self.heads))?
            // Move heads ahead of query and key positions, then add the shared batch axis.
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            .permute((2, 0, 1))?
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            .unsqueeze(0)?)
    }
}
