//! A checkpoint whose weights are one GGUF file, as llama.cpp publishes a
//! fine-tune (Jeden's goal model is published this way and no other).
//!
//! The decoder is Ster's own: each tensor is read out of the file, dequantized
//! to the dtype the run maps at, and handed to the decoder under the
//! Transformers name it asks for, so steering, capture and decisions run on a
//! GGUF exactly as on safetensors. The config and the tokenizer are the base
//! model's, laid beside the file in the checkpoint directory: a GGUF carries
//! its own vocabulary, but Ster reads the Transformers files every other
//! checkpoint has, so the two can never disagree on what a token is.
//!
//! The names follow llama.cpp's `convert_hf_to_gguf.py` for the dense Llama
//! and Qwen families (`blk.{i}.attn_q`, `ffn_gate`, `token_embd`). Those
//! families' query and key rows are stored unpermuted, which is what lets the
//! decoder read them directly; a name with no GGUF counterpart is asked for as
//! it is, and its absence is then reported by name.

use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::{Mutex, PoisonError},
};

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor, quantized::gguf_file};
use candle_nn::{Init, var_builder::SimpleBackend};

use crate::failures::{
    file,
    model::weights::{GGUF_TENSOR_UNREADABLE, GGUF_TRUNCATED, GGUF_UNREADABLE},
    refuse, refuse_candle,
};

/// One GGUF file, its header read once, its tensors read on demand.
pub struct GgufWeights {
    path: PathBuf,
    content: gguf_file::Content,
    reader: Mutex<BufReader<File>>,
}

impl GgufWeights {
    pub fn open(path: &Path) -> Result<Self> {
        let opened = File::open(path)
            .map_err(|error| file::read::refusal("the GGUF weights", path, &error))?;
        let mut reader = BufReader::new(opened);
        let content = gguf_file::Content::read(&mut reader).map_err(|error| {
            let cause = match file::tensors::io(&error) {
                Some(io) if io.kind() == std::io::ErrorKind::UnexpectedEof => &GGUF_TRUNCATED,
                Some(io) => return file::read::refusal("the GGUF weights", path, io),
                None => &GGUF_UNREADABLE,
            };
            refuse(
                cause,
                &[
                    ("path", &path.display().to_string()),
                    ("state", &file::state(path)),
                    ("error", &error.to_string()),
                ],
            )
        })?;
        Ok(Self {
            path: path.to_owned(),
            content,
            reader: Mutex::new(reader),
        })
    }

    /// The tensor the decoder names `name`, dequantized on `device`. One the
    /// file does not list is Candle's `CannotFindTensor`, which the load
    /// refuses as weights that do not fit, naming it; one it lists and
    /// cannot read is the file unreadable.
    fn load(&self, name: &str, device: &Device) -> candle_core::Result<Tensor> {
        let stored = stored_name(name);
        if !self.content.tensor_infos.contains_key(&stored) {
            return Err(candle_core::Error::CannotFindTensor { path: stored }.bt());
        }
        // A read seeks to the tensor's own offset before it reads, so a
        // reader another thread left mid-read by panicking is whole again.
        let mut reader = self.reader.lock().unwrap_or_else(PoisonError::into_inner);
        let quantized = self
            .content
            .tensor(&mut *reader, &stored, device)
            .map_err(|error| {
                refuse_candle(
                    &GGUF_TENSOR_UNREADABLE,
                    &[
                        ("path", &self.path.display().to_string()),
                        ("state", &file::state(&self.path)),
                        ("tensor", &stored),
                        ("name", name),
                        ("error", &error.to_string()),
                    ],
                )
            })?;
        quantized.dequantize(device)
    }
}

impl SimpleBackend for GgufWeights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load(name, device)?;
        if tensor.shape() != &shape {
            return Err(crate::model::misshapen(&stored_name(name), shape, tensor.shape()));
        }
        tensor.to_dtype(dtype)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.load(name, device)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(&stored_name(name))
    }
}

/// The GGUF name of a tensor asked for by its Transformers name; a name with
/// no GGUF counterpart is returned as it is.
pub fn stored_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".to_owned(),
        "model.norm.weight" => return "output_norm.weight".to_owned(),
        "lm_head.weight" => return "output.weight".to_owned(),
        _ => {}
    }
    let Some((layer, inner)) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
    else {
        return name.to_owned();
    };
    let Some((module, leaf)) = inner.rsplit_once('.') else {
        return name.to_owned();
    };
    let renamed = match module.split_once('.') {
        None => match module {
            "input_layernorm" => Some("attn_norm"),
            "post_attention_layernorm" => Some("ffn_norm"),
            _ => None,
        },
        Some(("self_attn", part)) => attention_name(part),
        Some(("mlp", projection)) => feed_forward_name(projection),
        Some(_) => None,
    };
    match renamed {
        Some(module) => format!("blk.{layer}.{module}.{leaf}"),
        None => name.to_owned(),
    }
}

fn attention_name(part: &str) -> Option<&'static str> {
    Some(match part {
        "q_proj" => "attn_q",
        "k_proj" => "attn_k",
        "v_proj" => "attn_v",
        "o_proj" => "attn_output",
        "q_norm" => "attn_q_norm",
        "k_norm" => "attn_k_norm",
        _ => return None,
    })
}

fn feed_forward_name(projection: &str) -> Option<&'static str> {
    Some(match projection {
        "gate_proj" => "ffn_gate",
        "up_proj" => "ffn_up",
        "down_proj" => "ffn_down",
        _ => return None,
    })
}
