//! The embeddings some families add beside the token table: Gemma 4's
//! per-layer inputs and LongCat-Flash-Lite's n-gram tables.

use candle_core::{DType, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding};

use crate::failures::{model::decoder::NGRAM_ROWS, refuse_candle};

use super::super::{Cache, Mode, NgramSpec, layer::norm::Norm};

/// Gemma 4's per-layer input tables, below the model root:
/// `embed_tokens_per_layer`, `per_layer_model_projection` and
/// `per_layer_projection_norm`.
#[derive(Debug, Clone)]
pub(super) struct PerLayerEmbeddings {
    pub(super) embeddings: Embedding,
    pub(super) projection: Linear,
    pub(super) norm: Norm,
    pub(super) width: usize,
}

impl PerLayerEmbeddings {
    /// Every layer's input, `[batch, sequence, layers, width]`: the token's
    /// row of `embed_tokens_per_layer` times `sqrt(width)`, plus the scaled
    /// embeddings projected, times `hidden_size^-0.5` and normed, the sum
    /// times `2^-0.5`, as Transformers' `get_per_layer_inputs` and
    /// `project_per_layer_inputs` compute them.
    pub(super) fn inputs(
        &self,
        tokens: &Tensor,
        embedded: &Tensor,
        layers: usize,
        mode: Mode,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence) = tokens.dims2()?;
        let hidden = embedded.dim(candle_core::D::Minus1)?;
        let shape = (batch, sequence, layers, self.width);
        let identity =
            (self.embeddings.forward(tokens)? * (self.width as f64).sqrt())?.reshape(shape)?;
        let context =
            (self.projection.forward(embedded)? * (hidden as f64).powf(-0.5))?.reshape(shape)?;
        let context = self.norm.forward(&context, mode.pass)?;
        (context + identity)? * std::f64::consts::FRAC_1_SQRT_2
    }
}

/// LongCat's n-gram tables, below the model root: `embedders.{i}` and
/// `post_projs.{i}` for each of `splits · (neighbors - 1)` tables, as
/// LongCat-Flash-Lite's `modeling_longcat_ngram.py` names them.
#[derive(Debug, Clone)]
pub(super) struct NgramEmbeddings {
    tables: Vec<(Embedding, Linear)>,
    spec: NgramSpec,
    vocab: u64,
}

impl NgramEmbeddings {
    pub(super) fn load(
        spec: NgramSpec,
        vocab: usize,
        hidden: usize,
        builder: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        let count = spec.splits * spec.neighbors.saturating_sub(1);
        // `longcat_ngram`, the one reading that states an `NgramSpec`,
        // refuses `splits == 0 || neighbors < 2` and `hidden % tables != 0`.
        assert!(
            count > 0 && hidden % count == 0,
            "longcat_ngram refuses n-gram tables that do not split hidden_size"
        );
        let width = hidden / count;
        let tables = (0..count)
            .map(|index| -> candle_core::Result<(Embedding, Linear)> {
                Ok((
                    embedding(
                        Self::rows(spec, vocab, index),
                        width,
                        builder.pp(format!("embedders.{index}")),
                    )?,
                    candle_nn::linear_no_bias(
                        width,
                        hidden,
                        builder.pp(format!("post_projs.{index}")),
                    )?,
                ))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            tables,
            spec,
            vocab: vocab as u64,
        })
    }

    /// Table `index`'s height, which is also the modulus its hashes reduce by.
    fn rows(spec: NgramSpec, vocab: usize, index: usize) -> usize {
        spec.ratio * vocab + 2 * index + 1
    }

    /// `embedded` (the token embeddings of `tokens`) averaged with every
    /// table's projected row, the n-grams reaching back into the tokens the
    /// cache kept from the previous call when this call continues one.
    /// Hashing follows `NgramEmbedding.forward`: for order `i` the id is
    /// `token + Σ_{d=1}^{i-1} token_{-d} · vocab^d mod rows`, reduced mod
    /// `rows`, a token before the start of its run (the sequence, or the
    /// last `eos` up to and including it) counting as zero.
    pub(super) fn embed(
        &self,
        tokens: &Tensor,
        embedded: Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence) = tokens.dims2()?;
        let rows: Vec<Vec<u32>> = tokens.to_dtype(DType::U32)?.to_vec2()?;
        let kept = self.spec.neighbors - 1;
        let earlier = cache
            .ngram_context
            .take()
            .filter(|_| cache.use_kv_cache && index_pos > 0);
        let contexts: Vec<Vec<u32>> = rows
            .iter()
            .enumerate()
            .map(|(row, tokens)| -> candle_core::Result<Vec<u32>> {
                // The first call of a decode has no earlier tokens.
                let mut context = match &earlier {
                    Some(earlier) => match earlier.get(row) {
                        Some(kept) => kept.clone(),
                        None => {
                            return Err(refuse_candle(
                                &NGRAM_ROWS,
                                &[
                                    ("row", &row.to_string()),
                                    ("kept", &earlier.len().to_string()),
                                ],
                            ));
                        }
                    },
                    None => Vec::new(),
                };
                context.extend_from_slice(tokens);
                Ok(context)
            })
            .collect::<candle_core::Result<_>>()?;
        if cache.use_kv_cache {
            cache.ngram_context = Some(
                contexts
                    .iter()
                    .map(|context| context[context.len().saturating_sub(kept)..].to_vec())
                    .collect(),
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
                sum = (sum
                    + projection
                        .forward(&table.forward(&ids)?)?
                        .to_dtype(embedded.dtype())?)?;
            }
        }
        sum / (1 + self.tables.len()) as f64
    }
}

