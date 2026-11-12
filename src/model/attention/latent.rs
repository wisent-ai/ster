//! Multi-head latent attention's projections: DeepSeek's low-rank query and
//! key-value bottlenecks.

use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

use crate::lora::Adapter;

use super::super::{
    IndexerSpec, LatentAttention, Mode,
    layer::{
        norm::{Norm, NormSpec},
        projection,
    },
};
use super::{indexer::Indexer, project};

/// Multi-head latent attention's projections, by DeepSeek's names below the
/// attention block: `q_proj`, or `q_a_proj`, `q_a_layernorm` and `q_b_proj`;
/// `kv_a_proj_with_mqa`, `kv_a_layernorm` and `kv_b_proj`.
#[derive(Debug, Clone)]
pub(super) struct Latent {
    query_down: Option<QueryDown>,
    query_up: Linear,
    key_value_down: Linear,
    key_value_norm: Norm,
    key_value_up: Linear,
    /// Dots3-Note's `k_rope_only_layernorm` over the shared rotated key part.
    rotated_norm: Option<Norm>,
    spec: LatentAttention,
}

/// The query bottleneck `q_a_layernorm(q_a_proj(x))`, and DeepSeek Sparse
/// Attention's indexer, which reads the latent query it makes: a latent
/// attention without the bottleneck has no latent query for an indexer, and
/// the config reading refuses an indexer there (`sparse_index`).
#[derive(Debug, Clone)]
struct QueryDown {
    down: Linear,
    norm: Norm,
    indexer: Option<Indexer>,
}

/// The indexer of a layer that has one and the latent query it reads.
pub(super) type Indexed<'a> = (&'a Indexer, Tensor);

impl Latent {
    /// `scales` multiply the query and key-value bottlenecks' normed
    /// outputs (LongCat-Flash's `mla_scale_q_lora`, `mla_scale_kv_lora`;
    /// Dots3-Note's `apply_mla_qkv_lora_rescale`); `rotated_norm` loads
    /// Dots3-Note's `k_rope_only_layernorm`; `indexer` is this layer's
    /// DeepSeek Sparse Attention indexer, which the query bottleneck feeds.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        input: usize,
        heads: usize,
        spec: LatentAttention,
        bias: bool,
        norms: NormSpec,
        scales: Option<(f64, f64)>,
        rotated_norm: bool,
        indexer: Option<IndexerSpec>,
    ) -> candle_core::Result<Self> {
        let query_width = heads * (spec.unrotated + spec.rotated);
        let (query_down, query_up) = match spec.query_rank {
            Some(rank) => (
                Some(QueryDown {
                    down: projection(input, rank, bias, false, builder.pp("q_a_proj"))?,
                    norm: scaled_by(
                        norms.load(rank, builder.pp("q_a_layernorm"))?,
                        scales.map(|(query, _)| query),
                    )?,
                    indexer: indexer
                        .map(|index| Indexer::load(builder, input, rank, index))
                        .transpose()?,
                }),
                projection(rank, query_width, false, false, builder.pp("q_b_proj"))?,
            ),
            None => (
                None,
                projection(input, query_width, false, false, builder.pp("q_proj"))?,
            ),
        };
        Ok(Self {
            query_down,
            query_up,
            key_value_down: projection(
                input,
                spec.key_value_rank + spec.rotated,
                bias,
                false,
                builder.pp("kv_a_proj_with_mqa"),
            )?,
            key_value_norm: scaled_by(
                norms.load(spec.key_value_rank, builder.pp("kv_a_layernorm"))?,
                scales.map(|(_, key_value)| key_value),
            )?,
            key_value_up: projection(
                spec.key_value_rank,
                heads * (spec.unrotated + spec.value),
                false,
                false,
                builder.pp("kv_b_proj"),
            )?,
            rotated_norm: if rotated_norm {
                Some(norms.load(spec.rotated, builder.pp("k_rope_only_layernorm"))?)
            } else {
                None
            },
            spec,
        })
    }

    /// Query, key and value as `[batch, sequence, heads, width]`.
    ///
    /// Each head's query and key are laid out rotated part first, then the
    /// position-free part: DeepSeek concatenates them the other way round,
    /// but a dot product does not see a permutation applied to both sides,
    /// and rotated-first is the layout the shared rotation code rotates (the
    /// first `rotary_dim` components). The rotated key part comes from the
    /// key-value bottleneck once and is shared by every head.
    pub(super) fn forward(
        &self,
        hidden: &Tensor,
        heads: usize,
        query_adapter: Option<&Adapter>,
        mode: Mode,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Option<Indexed<'_>>)> {
        let (batch, sequence, _) = hidden.dims3()?;
        let LatentAttention {
            key_value_rank,
            unrotated,
            rotated,
            value,
            ..
        } = self.spec;
        // The latent query `q_a_layernorm(q_a_proj(x))` is returned with
        // DeepSeek Sparse Attention's indexer, which reads it.
        let (query, indexed) = match &self.query_down {
            Some(bottleneck) => {
                let latent_query = bottleneck
                    .norm
                    .forward(&bottleneck.down.forward(hidden)?, mode.pass)?;
                let query = self.query_up.forward(&latent_query)?;
                let indexed = bottleneck
                    .indexer
                    .as_ref()
                    .map(|indexer| (indexer, latent_query));
                (query, indexed)
            }
            None => (
                project(&self.query_up, query_adapter, hidden, mode.route)?,
                None,
            ),
        };
        let query = query.reshape((batch, sequence, heads, unrotated + rotated))?;
        // A latent attention with no rotated part (GLM-5-Next's
        // `qk_rope_head_dim` 0) is its position-free part alone.
        if rotated == 0 {
            let compressed = self.key_value_down.forward(hidden)?;
            let expanded = self
                .key_value_up
                .forward(&self.key_value_norm.forward(
                    &compressed.narrow(2, 0, key_value_rank)?.contiguous()?,
                    mode.pass,
                )?)?
                .reshape((batch, sequence, heads, unrotated + value))?;
            let key = expanded.narrow(3, 0, unrotated)?.contiguous()?;
            let value = expanded.narrow(3, unrotated, value)?.contiguous()?;
            return Ok((query, key, value, indexed));
        }
        let query = Tensor::cat(
            &[
                &query.narrow(3, unrotated, rotated)?,
                &query.narrow(3, 0, unrotated)?,
            ],
            3,
        )?;
        let compressed = self.key_value_down.forward(hidden)?;
        let latent = compressed.narrow(2, 0, key_value_rank)?.contiguous()?;
        let shared = compressed
            .narrow(2, key_value_rank, rotated)?
            .contiguous()?;
        let shared = match &self.rotated_norm {
            Some(norm) => norm.forward(&shared, mode.pass)?,
            None => shared,
        };
        let shared = shared
            .unsqueeze(2)?
            .broadcast_as((batch, sequence, heads, rotated))?
            .contiguous()?;
        let expanded = self
            .key_value_up
            .forward(&self.key_value_norm.forward(&latent, mode.pass)?)?
            .reshape((batch, sequence, heads, unrotated + value))?;
        let key = Tensor::cat(&[&shared, &expanded.narrow(3, 0, unrotated)?], 3)?;
        let value = expanded.narrow(3, unrotated, value)?.contiguous()?;
        Ok((query, key, value, indexed))
    }
}

/// `norm` with its output multiplied by `scale`, or as it is without one.
fn scaled_by(norm: Norm, scale: Option<f64>) -> candle_core::Result<Norm> {
    match scale {
        Some(scale) => norm.scaled(scale),
        None => Ok(norm),
    }
}
