//! A T5 checkpoint's `config.json`, read into the dimensions the encoder and
//! decoder stacks are built from. Every dimension is the checkpoint's own; a
//! key Transformers' `T5Config` gives a default is read as that default when
//! the config leaves it out, and a key it needs that has no default is refused
//! by name.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::model::{Activation, NormKind, layer::norm::NormSpec};

/// `T5Config.relative_attention_num_buckets` when a config leaves it out:
/// https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/configuration_t5.py
const DEFAULT_BUCKETS: usize = 32;
/// `T5Config.relative_attention_max_distance` when a config leaves it out:
/// https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/configuration_t5.py
const DEFAULT_MAX_DISTANCE: usize = 128;
/// `T5Config.layer_norm_epsilon` when a config leaves it out:
/// https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/configuration_t5.py
const DEFAULT_NORM_EPS: f64 = 1e-6;

/// The feed-forward a `feed_forward_proj` names: `gated-<act>` (T5 v1.1,
/// FLAN-T5) multiplies `wi_0`'s activation by `wi_1`, a bare `<act>` (the
/// original T5) runs `wi` through it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FeedForward {
    pub gated: bool,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Config {
    pub vocab: usize,
    /// `d_model`, the residual width of both stacks.
    pub hidden: usize,
    pub heads: usize,
    /// `d_kv`, one head's width.
    pub head_width: usize,
    /// `num_heads · d_kv`, which T5 does not tie to `d_model`.
    pub attention_width: usize,
    /// `d_ff`.
    pub intermediate: usize,
    pub encoder_layers: usize,
    /// `num_decoder_layers`, else `num_layers`.
    pub decoder_layers: usize,
    pub buckets: usize,
    pub max_distance: usize,
    pub norm_eps: f64,
    pub feed_forward: FeedForward,
    /// `tie_word_embeddings`: the head is the shared embedding.
    pub tied_head: bool,
    /// A tied head reads the decoder's output multiplied by `d_model^-0.5`
    /// (`T5ForConditionalGeneration.forward`); an untied one reads it as is.
    pub scale_decoder_outputs: bool,
    /// `n_positions` when the config states it.
    pub context_length: Option<usize>,
    /// The token the decoder starts from (`decoder_start_token_id`).
    pub decoder_start: u32,
    pub pad: u32,
    pub eos: Vec<u32>,
}

impl Config {
    /// Reads `raw`, the `config.json` at `path`, with the feed-forward its
    /// `feed_forward_proj` names, which the loader resolves through the
    /// activation names every family shares.
    pub(crate) fn read(raw: &Value, feed_forward: FeedForward, path: &Path) -> Result<Self> {
        let stated = |key: &str| raw.get(key).and_then(Value::as_u64).map(|value| value as usize);
        let size = |key: &str| -> Result<usize> {
            stated(key)
                .and_then(std::num::NonZeroUsize::new)
                .map(std::num::NonZeroUsize::get)
                .with_context(|| format!("{} declares a T5 model without {key}", path.display()))
        };
        let token = |key: &str| -> Result<u32> {
            raw.get(key)
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .with_context(|| format!("{} declares a T5 model without {key}", path.display()))
        };
        let (hidden, heads, head_width) = (size("d_model")?, size("num_heads")?, size("d_kv")?);
        let encoder_layers = size("num_layers")?;
        let tied_head = raw
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .is_none_or(|tied| tied);
        let eos = match raw.get("eos_token_id") {
            Some(Value::Array(ids)) => ids
                .iter()
                .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                .collect::<Option<Vec<u32>>>()
                .with_context(|| {
                    format!("{} lists an eos_token_id that is not a token id", path.display())
                })?,
            _ => vec![token("eos_token_id")?],
        };
        if eos.is_empty() {
            bail!("{} lists no eos_token_id", path.display());
        }
        Ok(Self {
            vocab: size("vocab_size")?,
            hidden,
            heads,
            head_width,
            attention_width: heads * head_width,
            intermediate: size("d_ff")?,
            encoder_layers,
            decoder_layers: match stated("num_decoder_layers") {
                Some(layers) => layers,
                None => encoder_layers,
            },
            buckets: match stated("relative_attention_num_buckets") {
                Some(buckets) => buckets,
                None => DEFAULT_BUCKETS,
            },
            max_distance: match stated("relative_attention_max_distance") {
                Some(distance) => distance,
                None => DEFAULT_MAX_DISTANCE,
            },
            norm_eps: match raw.get("layer_norm_epsilon").and_then(Value::as_f64) {
                Some(eps) => eps,
                None => DEFAULT_NORM_EPS,
            },
            feed_forward,
            tied_head,
            scale_decoder_outputs: tied_head,
            context_length: stated("n_positions"),
            decoder_start: token("decoder_start_token_id")?,
            pad: token("pad_token_id")?,
            eos,
        })
    }

    /// T5's layer norm: a root mean square with a scale, no mean, no bias.
    pub(crate) fn norm(&self) -> NormSpec {
        NormSpec {
            kind: NormKind::Rms,
            eps: self.norm_eps,
            offset: false,
            groups: std::num::NonZeroUsize::MIN.get(),
        }
    }
}
