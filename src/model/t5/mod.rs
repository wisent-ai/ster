//! Native T5 encoder-decoder model. Layer selections and adapters address the
//! decoder stack; source encoding and decoder cross-attention stay frozen.

use anyhow::{Context, Result};
use candle_nn::{Embedding, Linear, VarBuilder, embedding, linear_no_bias};

use crate::{lora::Adapters, model::layer::norm::Norm};

mod attention;
mod block;
mod cache;
pub(crate) mod config;
mod forward;

pub(crate) use cache::Cache;

use attention::relative::RelativeBias;
use block::Block;
use config::Config;

#[derive(Debug, Clone)]
pub(crate) struct Model {
    config: Config,
    embeddings: Embedding,
    encoder: Vec<Block>,
    decoder: Vec<Block>,
    encoder_bias: RelativeBias,
    decoder_bias: RelativeBias,
    encoder_norm: Norm,
    decoder_norm: Norm,
    head: Linear,
    adapters: Adapters,
}

impl Model {
    pub fn load(builder: VarBuilder<'_>, config: Config, adapters: Adapters) -> Result<Self> {
        let embeddings = embedding(config.vocab, config.hidden, builder.pp("shared"))?;
        let head = if config.tied_head {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden, config.vocab, builder.pp("lm_head"))?
        };
        let bias = |stack: &str, bidirectional| -> Result<RelativeBias> {
            let weight = builder
                .pp(stack)
                .pp("block.0.layer.0.SelfAttention.relative_attention_bias")
                .get((config.buckets, config.heads), "weight")?;
            RelativeBias::new(weight, config.max_distance, bidirectional)
        };
        let encoder_bias = bias("encoder", true)?;
        let decoder_bias = bias("decoder", false)?;
        let norm = config.norm();
        let encoder_norm = norm.load(config.hidden, builder.pp("encoder.final_layer_norm"))?;
        let decoder_norm = norm.load(config.hidden, builder.pp("decoder.final_layer_norm"))?;
        let encoder = load_stack(&builder, &config, false, &adapters)?;
        let decoder = load_stack(&builder, &config, true, &adapters)?;
        Ok(Self {
            config,
            embeddings,
            encoder,
            decoder,
            encoder_bias,
            decoder_bias,
            encoder_norm,
            decoder_norm,
            head,
            adapters,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn adapters(&self) -> &Adapters {
        &self.adapters
    }
}

fn load_stack(
    builder: &VarBuilder<'_>,
    config: &Config,
    decoder: bool,
    adapters: &Adapters,
) -> Result<Vec<Block>> {
    let (name, count) = if decoder {
        ("decoder", config.decoder_layers)
    } else {
        ("encoder", config.encoder_layers)
    };
    let blocks = builder.pp(name).pp("block");
    // Stored block indices enumerate each stack independently.
    // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
    (0..count)
        .map(|index| {
            Block::load(blocks.pp(index.to_string()), config, index, decoder, adapters)
                .with_context(|| format!("failed to load T5 {name} block {index}"))
        })
        .collect()
}
