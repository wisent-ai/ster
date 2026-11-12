//! Loading DeepSeek-V4's attention: the latent query, the shared key-value
//! head, the grouped output projection, the sinks, and the compressor and
//! indexer a compressed or sparse layer reaches past its window with.

use candle_core::DType;
use candle_nn::{VarBuilder, linear_no_bias};

use super::{CompressedAttention, Reach};
use crate::model::{
    CompressedSpec, NormKind,
    cache::{RotaryTable, base_frequencies, yarn_frequencies},
    deepseek4::{compressor::Compressor, indexer::Indexer},
    layer::norm::NormSpec,
};

impl CompressedAttention {
    /// `builder` is the layer's.
    pub(in crate::model) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: &CompressedSpec,
        eps: f64,
        layer: usize,
    ) -> candle_core::Result<Self> {
        let block = builder.pp("self_attn");
        let at = |set: u128| set >> layer & 1 == 1;
        let rms = NormSpec {
            kind: NormKind::Rms,
            eps,
            offset: false,
            groups: 1,
        };
        let width = spec.heads * spec.head_dim;
        let (reach, theta) = if at(spec.sparse_layers) {
            let compressor = block.pp("compressor");
            (
                Reach::Sparse(
                    Compressor::load(
                        &compressor,
                        hidden,
                        spec.head_dim,
                        spec.sparse_rate,
                        true,
                        eps,
                        0,
                    )?,
                    Box::new(Indexer::load(
                        &compressor.pp("indexer"),
                        hidden,
                        spec.query_rank,
                        (
                            spec.index_heads,
                            spec.index_head_dim,
                            spec.index_top_k,
                            spec.sparse_rate,
                        ),
                        eps,
                    )?),
                ),
                spec.compress_theta,
            )
        } else if at(spec.heavy_layers) {
            let compressor = block.pp("compressor");
            (
                Reach::Heavy(Compressor::load(
                    &compressor,
                    hidden,
                    spec.head_dim,
                    spec.heavy_rate,
                    false,
                    eps,
                    0,
                )?),
                spec.compress_theta,
            )
        } else {
            (Reach::Window, spec.main_theta)
        };
        let base = base_frequencies(spec.rotary_dim, theta);
        let frequencies = match (&reach, spec.compress_yarn) {
            (Reach::Window, _) | (_, None) => base,
            (_, Some(yarn)) => yarn_frequencies(
                &base,
                spec.rotary_dim,
                theta,
                yarn.factor,
                yarn.original,
                (yarn.beta_fast, yarn.beta_slow),
                yarn.truncate,
            ),
        };
        let groups = spec.output_groups;
        Ok(Self {
            query_down: linear_no_bias(hidden, spec.query_rank, block.pp("q_a_proj"))?,
            query_norm: rms.load(spec.query_rank, block.pp("q_a_norm"))?,
            query_up: linear_no_bias(spec.query_rank, width, block.pp("q_b_proj"))?,
            key_value: linear_no_bias(hidden, spec.head_dim, block.pp("kv_proj"))?,
            key_value_norm: rms.load(spec.head_dim, block.pp("kv_norm"))?,
            output_down: block
                .pp("o_a_proj")
                .get((groups * spec.output_rank, width / groups), "weight")?
                .reshape((groups, spec.output_rank, width / groups))?,
            output_up: linear_no_bias(groups * spec.output_rank, hidden, block.pp("o_b_proj"))?,
            sinks: block
                .get(spec.heads, "sinks")?
                .to_dtype(DType::F32)?
                .reshape((1, spec.heads, 1, 1))?,
            reach,
            rotation: RotaryTable::new(frequencies, 1.0, builder.device())?,
            heads: spec.heads,
            head_dim: spec.head_dim,
            groups,
            window: spec.window,
            eps,
        })
    }
}
