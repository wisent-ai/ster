//! A T5 checkpoint's `config.json`, read into the dimensions the encoder and
//! decoder stacks are built from. Every dimension is the checkpoint's own; a
//! key Transformers' `T5Config` gives a default is read as that default when
//! the config leaves it out, and a key it needs that has no default is refused
//! by name.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::failures::model::{Unrunnable, config::KEY_ABSENT, settings, stated as state_of};
use crate::failures::refuse;
use crate::model::keys::{Kind, flag_or, mistyped, number_or, stated, whole, whole_or};
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
    /// activation names every family shares. Every key goes through the
    /// shared reader, so one stated with another type is
    /// `ster.model.config.key-mistyped` and one left out that has no default
    /// is `key-absent`.
    pub(crate) fn read(raw: &Value, feed_forward: FeedForward, path: &Path) -> Result<Self> {
        let absent = |key: &str, kind: &str| {
            refuse(
                &KEY_ABSENT,
                &[
                    ("path", &path.display().to_string()),
                    ("key", key),
                    ("spellings", ""),
                    ("model_type", "t5"),
                    ("kind", kind),
                ],
            )
        };
        let size = |key: &str| -> Result<usize> {
            match whole(raw, key, path)? {
                Some(size) if size > 0 => Ok(size),
                Some(_) => Err(settings(
                    Unrunnable::OutOfRange,
                    format!("{} sets {key} to 0; a T5 model needs it above zero", path.display()),
                    state_of(raw, &[key]),
                )),
                None => Err(absent(key, "a whole number above zero")),
            }
        };
        let token = |key: &str| -> Result<u32> {
            stated::<u32>(raw, key, path)?.ok_or_else(|| absent(key, "a token id"))
        };
        let (hidden, heads, head_width) = (size("d_model")?, size("num_heads")?, size("d_kv")?);
        let encoder_layers = size("num_layers")?;
        let tied_head = flag_or(raw, "tie_word_embeddings", DEFAULT_TIED_HEAD, path)?;
        let eos = match raw.get("eos_token_id") {
            Some(Value::Array(ids)) => {
                let mut eos = Vec::with_capacity(ids.len());
                for (index, listed) in ids.iter().enumerate() {
                    let shown = format!("eos_token_id[{index}]");
                    let Some(id) = u32::read(listed) else {
                        return Err(mistyped(&shown, listed, "a token id", path));
                    };
                    eos.push(id);
                }
                eos
            }
            // One id, or none: `token` reads the key as one id and refuses
            // another type or its absence.
            _ => vec![token("eos_token_id")?],
        };
        if eos.is_empty() {
            return Err(settings(
                Unrunnable::Incomplete,
                format!("{} lists no eos_token_id; the decoder stops at one", path.display()),
                state_of(raw, &["eos_token_id"]),
            ));
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
            decoder_layers: whole_or(raw, "num_decoder_layers", encoder_layers, path)?,
            buckets: whole_or(raw, "relative_attention_num_buckets", DEFAULT_BUCKETS, path)?,
            max_distance: whole_or(raw, "relative_attention_max_distance", DEFAULT_MAX_DISTANCE, path)?,
            norm_eps: number_or(raw, "layer_norm_epsilon", DEFAULT_NORM_EPS, path)?,
            feed_forward,
            tied_head,
            scale_decoder_outputs: tied_head,
            context_length: whole(raw, "n_positions", path)?,
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
