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
    sync::Mutex,
};

use anyhow::{Context, Result};
use candle_core::{quantized::gguf_file, DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init};

/// One GGUF file, its header read once, its tensors read on demand.
pub struct GgufWeights {
    path: PathBuf,
    content: gguf_file::Content,
    reader: Mutex<BufReader<File>>,
}

impl GgufWeights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("failed to open GGUF weights {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .with_context(|| format!("{} is not a GGUF file Ster can read", path.display()))?;
        Ok(Self { path: path.to_owned(), content, reader: Mutex::new(reader) })
    }

    /// The tensor the decoder names `name`, dequantized on `device`.
    fn load(&self, name: &str, device: &Device) -> candle_core::Result<Tensor> {
        let stored = stored_name(name);
        let Ok(mut reader) = self.reader.lock() else {
            candle_core::bail!("the reader of {} was left broken by an earlier failed read", self.path.display());
        };
        let quantized = self.content.tensor(&mut *reader, &stored, device).map_err(|error| {
            candle_core::Error::Msg(format!("{} holds no {stored} (asked for as {name}): {error}", self.path.display()))
        })?;
        quantized.dequantize(device)
    }
}

impl SimpleBackend for GgufWeights {
    fn get(&self, shape: Shape, name: &str, _: Init, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.load(name, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!(
                "{} stores {} with shape {:?}; the decoder's config expects {:?}",
                self.path.display(),
                stored_name(name),
                tensor.shape(),
                shape
            );
        }
        tensor.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
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
    let Some((layer, inner)) = name.strip_prefix("model.layers.").and_then(|rest| rest.split_once('.')) else {
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
