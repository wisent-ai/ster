//! Which decoder family a checkpoint is, read from its `config.json`.
//!
//! Every switch below is a key the checkpoint publishes; none is a value Ster
//! picks. A config that asks for something the decoder does not implement is
//! refused with the key that asked for it, instead of being loaded wrong.

use std::path::Path;

use anyhow::{Result, bail};
use candle_transformers::models::llama::LlamaConfig;
use serde_json::Value;

use crate::model::Architecture;
use crate::runtime::load::family::custom::hrm::hrm_text;
use crate::runtime::load::family::custom::inkling::inkling;
use crate::runtime::load::family::custom::llama4::llama4;
use crate::runtime::load::family::gemma::gemma3n::gemma3n;
use crate::runtime::load::family::gemma::laguna::laguna;
use crate::runtime::load::family::gemma::plamo::{muse_glimmer, plamo3};
use crate::runtime::load::family::latent::mimo::mimo_v2;
use crate::runtime::load::family::latent::sarvam::sarvam_mla;
use crate::runtime::load::family::latent::v4::deepseek_v4;
use crate::runtime::load::family::read::activation::activation;
use crate::runtime::load::family::read::layers::{every_layer, windowed_layers};
use crate::runtime::load::family::read::number;
use crate::runtime::load::family::rope::head::head;
use crate::runtime::load::family::sparse::afmoe::afmoe;
use crate::runtime::load::family::sparse::cohere2_moe::cohere2_moe;
use crate::runtime::load::family::sparse::glm5::glm5_next;
use crate::runtime::load::family::sparse::hy_v4::hy_v4;
use crate::runtime::load::family::sparse::longcat::{longcat_flash, longcat_ngram};
use crate::runtime::load::family::step::k2::k2_horizon;
use crate::runtime::load::family::step::step3p5::step3p5;

mod experts;
mod keys;
mod kind;
mod read;
mod rope;
mod dense {
    pub(in crate::runtime::load::family) mod granite;
    pub(in crate::runtime::load::family) mod llama;
    pub(in crate::runtime::load::family) mod olmo;
    pub(in crate::runtime::load::family) mod qwen;
    pub(in crate::runtime::load::family) mod small;
}
mod classic {
    pub(in crate::runtime::load::family) mod glm;
    pub(in crate::runtime::load::family) mod gpt;
    pub(in crate::runtime::load::family) mod hunyuan;
    pub(in crate::runtime::load::family) mod loops;
}
mod state {
    pub(in crate::runtime::load::family) mod falcon_h1;
    pub(in crate::runtime::load::family) mod hybrid;
    pub(in crate::runtime::load::family) mod mamba;
    pub(in crate::runtime::load::family) mod minimax;
    pub(in crate::runtime::load::family) mod zamba;
}
mod linear {
    pub(in crate::runtime::load::family) mod kda;
    pub(in crate::runtime::load::family) mod kimi;
    pub(in crate::runtime::load::family) mod ling;
}
mod latent {
    pub(in crate::runtime::load::family) mod deepseek;
    pub(in crate::runtime::load::family) mod dots3;
    pub(in crate::runtime::load::family) mod mimo;
    pub(in crate::runtime::load::family) mod sarvam;
    pub(in crate::runtime::load::family) mod v4;
}
mod sparse {
    pub(in crate::runtime::load::family) mod afmoe;
    pub(in crate::runtime::load::family) mod cohere2_moe;
    pub(in crate::runtime::load::family) mod glm5;
    pub(in crate::runtime::load::family) mod hy_v4;
    pub(in crate::runtime::load::family) mod longcat;
}
mod step {
    pub(in crate::runtime::load::family) mod k2;
    pub(in crate::runtime::load::family) mod step3p5;
}
mod gemma {
    pub(in crate::runtime::load::family) mod gemma;
    pub(in crate::runtime::load::family) mod gemma3n;
    pub(in crate::runtime::load::family) mod gemma4;
    pub(in crate::runtime::load::family) mod laguna;
    pub(in crate::runtime::load::family) mod plamo;
}
mod custom {
    pub(in crate::runtime::load::family) mod hrm;
    pub(in crate::runtime::load::family) mod inkling;
    pub(in crate::runtime::load::family) mod llama4;
    pub(in crate::runtime::load::family) mod minimax_m3;
}

pub(super) use keys::fill_llama_keys;
pub(super) use kind::Family;
pub(super) use read::candle::llama_config;
pub(super) use rope::take_rope_scaling;

/// What every family's reading reads: the config's `model_type`, its keys, the
/// `rope_scaling` [`take_rope_scaling`] took, Candle's parse of the Llama keys,
/// the layer count and the config's path, which every refusal names.
#[derive(Clone, Copy)]
pub(in crate::runtime::load::family) struct At<'a> {
    pub(in crate::runtime::load::family) model_type: &'a str,
    pub(in crate::runtime::load::family) raw: &'a Value,
    pub(in crate::runtime::load::family) scaling: Option<&'a Value>,
    pub(in crate::runtime::load::family) llama: &'a LlamaConfig,
    pub(in crate::runtime::load::family) layers: usize,
    pub(in crate::runtime::load::family) path: &'a Path,
}

/// What `model_type` adds to the Llama block, from the config's own keys and
/// the `rope_scaling` [`take_rope_scaling`] took out of them.
pub(super) fn family(
    model_type: &str,
    raw: &Value,
    scaling: Option<&Value>,
    llama: &LlamaConfig,
    path: &Path,
) -> Result<Architecture> {
    let layers = llama.num_hidden_layers;
    let mut architecture = Architecture::llama(
        llama.hidden_size,
        llama.num_attention_heads,
        llama.rms_norm_eps,
    );
    head(raw, scaling, llama, &mut architecture, path)?;
    // Gemma's configs name `gelu` but Transformers runs the tanh
    // approximation for every Gemma, so the family decides, not the key.
    if !model_type.starts_with("gemma") && model_type != "bloom" {
        architecture.activation = activation(raw, model_type, path)?;
    }
    let at = At {
        model_type,
        raw,
        scaling,
        llama,
        layers,
        path,
    };
    match model_type {
        "ministral3" | "llama" | "mistral" | "mixtral" | "phi3" | "phimoe" | "stablelm"
        | "starcoder2" | "cohere" | "cohere2" | "nemotron" | "phi" => {
            dense::llama::configure(&at, &mut architecture)?
        }
        "qwen2" | "qwen3" | "qwen2_moe" | "qwen3_moe" | "qwen3_next" | "mimo" | "mellum"
        | "qwen3_5_text" | "qwen3_5_moe_text" => dense::qwen::configure(&at, &mut architecture)?,
        "granite" | "granitemoe" | "granitemoehybrid" | "granite_swa" | "granitemoe_swa"
        | "granitemoeshared" | "hyperclovax" => dense::granite::configure(&at, &mut architecture)?,
        "olmoe" | "olmo_hybrid" | "olmo2" | "olmo3" | "flex_olmo" | "smollm3" | "olmo" | "dbrx"
        | "exaone4" | "apertus" => dense::olmo::configure(&at, &mut architecture)?,
        "nemotron-nas" | "nanbeige" | "iquestloopcoder" => {
            classic::loops::configure(&at, &mut architecture)?
        }
        "hy_v3" | "ernie4_5" | "ernie4_5_moe" | "gpt_oss" | "hunyuan_v1_dense"
        | "hunyuan_v1_moe" => classic::hunyuan::configure(&at, &mut architecture)?,
        "exaone_moe" | "internlm3" | "TeleFLM" | "seed_oss" | "solar" | "PanguEmbedded"
        | "telechat3" | "step1" | "iquestcoder" | "arcee" | "jais2" | "minicpm" | "orion" => {
            dense::small::configure(&at, &mut architecture)?
        }
        "mamba" | "falcon_mamba" | "jamba" | "mamba2" => {
            state::mamba::configure(&at, &mut architecture)?
        }
        "bamba" | "nemotron_h" | "nemotron_h_puzzle" => {
            state::hybrid::configure(&at, &mut architecture)?
        }
        "bailing_moe" | "param2moe" | "bailing_hybrid" | "sarvam_moe" => {
            linear::ling::configure(&at, &mut architecture)?
        }
        "falcon_h1" | "lfm2" | "lfm2_moe" => state::falcon_h1::configure(&at, &mut architecture)?,
        "glm4_moe" | "deepseek_v2" | "deepseek_v3" | "minicpm3" | "glm4_moe_lite"
        | "deepseek_v32" | "glm_moe_dsa" | "axk1" | "pangu_ultra_moe" | "dots3_note"
        | "step3_text" | "deepseek" => latent::deepseek::configure(&at, &mut architecture)?,
        "internlm2" | "exaone" | "telechat" | "mpt" | "chatglm" | "glm" | "glm4" => {
            classic::glm::configure(&at, &mut architecture)?
        }
        "step3p5" => step3p5(raw, layers, llama, &mut architecture, path)?,
        "k2_horizon" => k2_horizon(raw, layers, llama, &mut architecture, path)?,
        "longcat_flash" => longcat_flash(raw, llama, &mut architecture, path)?,
        "longcat_flash_ngram" => {
            longcat_flash(raw, llama, &mut architecture, path)?;
            architecture.ngram = Some(longcat_ngram(raw, path)?);
        }
        "sarvam_mla" => sarvam_mla(raw, layers, &mut architecture, path)?,
        "cohere2_moe" => cohere2_moe(raw, layers, &mut architecture, path)?,
        "llama4_text" => llama4(raw, layers, &mut architecture, path)?,
        "minimax_m3_vl_text" => custom::minimax_m3::family(raw, layers, &mut architecture, path)?,
        "afmoe" => afmoe(raw, layers, llama, &mut architecture, path)?,
        "laguna" => laguna(raw, scaling, layers, llama, &mut architecture, path)?,
        "muse_glimmer_text" => muse_glimmer(raw, layers, llama, &mut architecture, path)?,
        "plamo3" => plamo3(raw, layers, llama, &mut architecture, path)?,
        "zamba2" => state::zamba::configure(&at, &mut architecture)?,
        "minimax_text_01" | "minimax" | "minimax_m2" => {
            state::minimax::configure(&at, &mut architecture)?
        }
        "kimi_linear" => linear::kimi::configure(&at, &mut architecture)?,
        "gpt_neox" | "gptj" | "gpt2" | "gpt_bigcode" | "opt" | "bloom" | "falcon" => {
            classic::gpt::configure(&at, &mut architecture)?
        }
        "gemma" | "gemma2" | "gemma3_text" | "vaultgemma" => {
            gemma::gemma::configure(&at, &mut architecture)?
        }
        "gemma3n_text" => gemma3n(raw, layers, llama, &mut architecture, path)?,
        "hrm_text" => hrm_text(raw, layers, &mut architecture, path)?,
        "inkling_text" => inkling(raw, layers, &mut architecture, path)?,
        "deepseek_v4" => deepseek_v4(raw, scaling, layers, &mut architecture, path)?,
        "glm5_next_text" => glm5_next(raw, layers, &mut architecture, path)?,
        "hy_v4" => hy_v4(raw, layers, &mut architecture, path)?,
        "gemma4_text" | "gemma4_unified_text" => gemma::gemma4::configure(&at, &mut architecture)?,
        "mimo_v2_flash" | "mimo_v2" => {
            mimo_v2(raw, model_type, layers, llama, &mut architecture, path)?
        }
        other => bail!("model architecture {other:?} has no decoder in this Ster build"),
    }
    // A rotation stated per layer kind (Transformers 5's `rope_parameters`
    // by layer type) gives sliding-window layers their own base, whatever
    // the family, so a scaling on the full-attention rotation (Mellum's
    // YaRN) never reaches them.
    if architecture.local_rope_theta.is_none() && architecture.sliding_window.is_some() {
        architecture.local_rope_theta =
            number(raw, "rope_local_base_freq").map(|theta| theta as f32);
    }
    // A family with recurrent mixers (LFM2's convolutions, Qwen3-Next's,
    // Kimi-Linear's and OLMo Hybrid's delta rule, MiniMax's lightning
    // attention, Granite 4.0's Mamba-2) reads its `layer_types` as which
    // layers run which mixer, above, Inkling its `hybrid` and
    // `hybrid_sliding` layers and DeepSeek-V4 its compressed ones; every
    // other family's say which layers attend through the window.
    let mixers_listed = architecture.short_convolution.is_some()
        || architecture.delta_rule.is_some()
        || architecture.lightning.is_some()
        || architecture.state_space.is_some()
        || architecture.inkling.is_some()
        || architecture.compressed.is_some();
    let windows = raw
        .get("layer_types")
        .and_then(Value::as_array)
        .filter(|_| !mixers_listed);
    if let Some(types) = windows {
        architecture.sliding_layers =
            windowed_layers(types, layers, architecture.chunk_lookback.is_some(), path)?;
    }
    // Cohere 2's global layers apply no rotary embedding; so do EXAONE 4's
    // when the model mixes local and global layers at all.
    let hybrid_exaone =
        matches!(model_type, "exaone4" | "exaone_moe") && architecture.sliding_window.is_some();
    if model_type == "cohere2" || hybrid_exaone {
        architecture.unrotated_layers = every_layer(layers, path)? & !architecture.sliding_layers;
    }
    if architecture.sliding_layers != 0 && architecture.sliding_window.is_none() {
        bail!(
            "{} makes some layers sliding-window but declares no sliding_window",
            path.display()
        );
    }
    Ok(architecture)
}
