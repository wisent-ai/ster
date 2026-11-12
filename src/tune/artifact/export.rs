//! export.rs — a Ster adapter written as a PEFT LoRA adapter directory.
//!
//! A merged checkpoint is one way to hand a finished adapter to another tool;
//! the other is to keep it an adapter and write it in the layout the serving
//! ecosystem loads beside a base model (vLLM, `transformers` + `peft`,
//! llama.cpp's converter): `adapter_config.json` and `adapter_model.safetensors`.
//!
//! Nothing is recomputed. A Ster factor pair is `a: [rank, inputs]` and
//! `b: [outputs, rank]` applied as `x -> a -> b` and scaled by `alpha / rank`,
//! which is exactly PEFT's `lora_A.weight`, `lora_B.weight` and `lora_alpha / r`.
//! Only the names change, and the names come from the same per-architecture
//! placement `merge` uses, so a projection is named the way the base
//! checkpoint names it. A projection stored fused with others (Phi-3's
//! `qkv_proj`, GPT-NeoX) or transposed (GPT-2's `Conv1D`) has no PEFT module
//! of its own and is refused by name rather than written under a module PEFT
//! would apply to the wrong rows. Dropout is a training setting and an
//! exported adapter is for inference, so the config carries none.

use std::{collections::HashMap, fs, path::Path};

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::Serialize;
use serde_json::json;

use super::fit::Fit;
use super::merge::{copy, tensors_unwritable, unwritable};
use crate::failures::tune::rewrite::{
    FUSED_UNEXPORTABLE, GGUF_UNEXPORTABLE, REWARD_UNEXPORTABLE, TRANSPOSED_UNEXPORTABLE,
};
use crate::{lora, workflow};

/// The prefix PEFT puts before every adapted module's checkpoint path.
const PEFT_ROOT: &str = "base_model.model";

#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub model: String,
    pub model_revision: Option<String>,
    pub adapter: String,
    pub output: String,
    pub format: &'static str,
    pub rank: usize,
    pub alpha: f64,
    /// The module names PEFT's `target_modules` lists, as the base checkpoint names them.
    pub target_modules: Vec<String>,
    pub layers: Vec<usize>,
    pub tensors: usize,
    /// Every file the export wrote, relative to the output directory.
    pub files: Vec<String>,
}

/// Writes the adapter at `adapter`, trained for `model`, as a PEFT LoRA
/// adapter directory at `output`, with the base's own tokenizer beside it.
pub fn export_peft(
    model: &str,
    revision: Option<&str>,
    adapter: &Path,
    output: &Path,
) -> Result<ExportReport> {
    let device = Device::Cpu;
    let artifact = lora::Artifact::load(adapter, &device)?;
    let fit = Fit {
        step: "exporting",
        path: adapter,
        model,
    };
    fit.adapter(&artifact, &REWARD_UNEXPORTABLE)?;
    fit.trained_for(&artifact.model)?;
    let source = fit.source(revision, &GGUF_UNEXPORTABLE)?;
    let (config, architecture) = (&source.config, &source.architecture);
    architecture.check_targets(&artifact.targets, config, model)?;
    fit.width(artifact.hidden_size, config.hidden_size)?;
    fit.layers(&artifact.layers, config.num_hidden_layers)?;
    artifact.validate_widths(lora::Widths::for_decoder(config, architecture))?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut modules: Vec<String> = Vec::new();
    for &layer in &artifact.layers {
        for &target in &artifact.targets {
            let placement = fit.placement(&source, target, layer)?;
            fit.whole(
                &placement,
                (target, layer),
                &FUSED_UNEXPORTABLE,
                Some(&TRANSPOSED_UNEXPORTABLE),
            )?;
            // Every projection tensor name ends in `.weight`; its module is
            // the name without it.
            let module = placement.tensor.trim_end_matches(".weight");
            let (a_name, b_name) = lora::Adapter::tensor_names(layer, target);
            for (factor, name) in [("lora_A", &a_name), ("lora_B", &b_name)] {
                let tensor = fit.tensor(&artifact.tensors, "the adapter", name, target, layer)?;
                let tensor = tensor
                    .contiguous()
                    .map_err(|error| fit.unfit(target, layer, name, tensor.dims(), error))?;
                tensors.insert(format!("{PEFT_ROOT}.{module}.{factor}.weight"), tensor);
            }
            // The leaf is the last dotted part; a module with no dot is its own.
            let leaf = match module.rsplit_once('.') {
                Some((_, leaf)) => leaf,
                None => module,
            }
            .to_owned();
            if !modules.contains(&leaf) {
                modules.push(leaf);
            }
        }
    }

    fs::create_dir_all(output).map_err(|error| unwritable(output, output, &error))?;
    let weights = output.join("adapter_model.safetensors");
    candle_core::safetensors::save(&tensors, &weights)
        .map_err(|error| tensors_unwritable(output, &weights, &error))?;
    let model_revision = source
        .checkpoint
        .revision
        .clone()
        .or_else(|| artifact.model_revision.clone());
    let adapter_config = json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model,
        "revision": model_revision,
        "r": artifact.rank,
        "lora_alpha": artifact.alpha,
        "bias": "none",
        "target_modules": modules,
        "layers_to_transform": artifact.layers,
        "inference_mode": true,
    });
    let config_path = output.join("adapter_config.json");
    let text = crate::json::pretty(&adapter_config).into_bytes();
    fs::write(&config_path, text).map_err(|error| unwritable(output, &config_path, &error))?;
    // A served adapter is loaded beside the base's tokenizer; shipping the
    // base's own copy keeps the pair from drifting to another tokenizer.
    copy(output, &source.checkpoint.tokenizer, &output.join("tokenizer.json"))?;
    let files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.json",
    ]
    .map(str::to_owned)
    .to_vec();
    workflow::progress(format!(
        "wrote {} to {}",
        files.join(", "),
        output.display()
    ));
    Ok(ExportReport {
        model: model.to_owned(),
        model_revision,
        adapter: adapter.display().to_string(),
        output: output.display().to_string(),
        format: "peft",
        rank: artifact.rank,
        alpha: artifact.alpha,
        target_modules: modules,
        layers: artifact.layers.clone(),
        tensors: tensors.len(),
        files,
    })
}
