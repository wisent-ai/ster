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
/// `PretrainedConfig.tie_word_embeddings` when a config leaves it out:
/// https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/configuration_utils.py
const DEFAULT_TIED_HEAD: bool = true;

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
        // A key left out or written `null` is absent; one of another type is
        // refused, never read as absent.
        let stated = |key: &str| -> Result<Option<u64>> {
            match raw.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(value) => match value.as_u64() {
                    Some(number) => Ok(Some(number)),
                    None => bail!(
                        "{} sets {key} to {value}, not a whole number; correct it in the checkpoint's config.json",
                        path.display()
                    ),
                },
            }
        };
        let count = |key: &str| -> Result<Option<usize>> {
            match stated(key)? {
                Some(number) => Ok(Some(usize::try_from(number).with_context(|| {
                    format!(
                        "{} sets {key} to {number}, past this platform's sizes",
                        path.display()
                    )
                })?)),
                None => Ok(None),
            }
        };
        let size = |key: &str| -> Result<usize> {
            match count(key)? {
                Some(size) if size > 0 => Ok(size),
                Some(size) => bail!(
                    "{} sets {key} to {size}; a T5 model needs it above zero",
                    path.display()
                ),
                None => bail!("{} declares a T5 model without {key}", path.display()),
            }
        };
        let id = |key: &str, number: u64| -> Result<u32> {
            u32::try_from(number).with_context(|| {
                format!(
                    "{} sets {key} to {number}, past the u32 token ids",
                    path.display()
                )
            })
        };
        let token = |key: &str| -> Result<u32> {
            match stated(key)? {
                Some(number) => id(key, number),
                None => bail!("{} declares a T5 model without {key}", path.display()),
            }
        };
        let (hidden, heads, head_width) = (size("d_model")?, size("num_heads")?, size("d_kv")?);
        let encoder_layers = size("num_layers")?;
        let tied_head = match raw.get("tie_word_embeddings") {
            None | Some(Value::Null) => DEFAULT_TIED_HEAD,
            Some(Value::Bool(tied)) => *tied,
            Some(other) => bail!(
                "{} sets tie_word_embeddings to {other}, not true or false; correct it in the checkpoint's config.json",
                path.display()
            ),
        };
        let eos = match raw.get("eos_token_id") {
            Some(Value::Array(ids)) => {
                let mut eos = Vec::with_capacity(ids.len());
                for listed in ids {
                    let Some(number) = listed.as_u64() else {
                        bail!(
                            "{} lists an eos_token_id {listed}, not a token id; correct it in the checkpoint's config.json",
                            path.display()
                        );
                    };
                    eos.push(id("eos_token_id", number)?);
                }
                eos
            }
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
            // `T5Config` sets `num_decoder_layers` to `num_layers` when it is
            // left out:
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/configuration_t5.py
            decoder_layers: match count("num_decoder_layers")? {
                Some(layers) => layers,
                None => encoder_layers,
            },
            buckets: match count("relative_attention_num_buckets")? {
                Some(buckets) => buckets,
                None => DEFAULT_BUCKETS,
            },
            max_distance: match count("relative_attention_max_distance")? {
                Some(distance) => distance,
                None => DEFAULT_MAX_DISTANCE,
            },
            norm_eps: match raw.get("layer_norm_epsilon") {
                None | Some(Value::Null) => DEFAULT_NORM_EPS,
                Some(value) => match value.as_f64() {
                    Some(eps) => eps,
                    None => bail!(
                        "{} sets layer_norm_epsilon to {value}, not a number; correct it in the checkpoint's config.json",
                        path.display()
                    ),
                },
            },
            feed_forward,
            tied_head,
            scale_decoder_outputs: tied_head,
            context_length: count("n_positions")?,
            decoder_start: token("decoder_start_token_id")?,
            pad: token("pad_token_id")?,
            eos,
        })
    }

    /// T5's layer norm: a root mean square with a scale, no mean, no bias.
    pub(in crate::model) fn norm(&self) -> NormSpec {
        NormSpec {
            kind: NormKind::Rms,
            eps: self.norm_eps,
            offset: false,
            // `T5LayerNorm` normalizes each position's whole hidden state as
            // one group:
            // https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/t5/modeling_t5.py
            groups: 1,
        }
    }
}
