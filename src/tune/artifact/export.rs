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

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use serde::Serialize;
use serde_json::json;

use crate::{
    lora,
    runtime::{Checkpoint, Layout},
    workflow,
};

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
    if artifact.kind != lora::Kind::Adapter {
        bail!(
            "adapter artifact is a {} model, not a generation adapter; PEFT has no place for its head",
            artifact.kind.name()
        );
    }
    if artifact.model != model {
        bail!(
            "adapter was trained for model {:?}, current model is {:?}",
            artifact.model,
            model
        );
    }
    let source = Checkpoint::resolve(model, revision)?;
    if source.layout == Layout::Gguf {
        bail!(
            "{model} is a GGUF checkpoint; a PEFT adapter is named after the base model's safetensors release, so export against that release"
        );
    }
    let (config, architecture, _) = source.decoder_config()?;
    architecture.check_targets(&artifact.targets, &config)?;
    artifact.validate_widths(lora::Widths::for_decoder(&config, &architecture))?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut modules: Vec<String> = Vec::new();
    for &layer in &artifact.layers {
        for &target in &artifact.targets {
            let placement = architecture
                .placement(target, layer, &config)
                .with_context(|| {
                    format!("this model has no {} projection to export", target.name())
                })?;
            if placement.blocks.is_some() || placement.transposed {
                bail!(
                    "this model stores the {} projection {} with others, so PEFT has no module of its own for it; merge the adapter (ster tune merge) instead",
                    target.name(),
                    if placement.transposed {
                        "transposed"
                    } else {
                        "fused"
                    }
                );
            }
            let module = placement.tensor.strip_suffix(".weight").with_context(|| {
                format!(
                    "checkpoint tensor {} is not a projection weight",
                    placement.tensor
                )
            })?;
            let (a_name, b_name) = lora::Adapter::tensor_names(layer, target);
            for (factor, name) in [("lora_A", &a_name), ("lora_B", &b_name)] {
                let tensor = artifact
                    .tensors
                    .get(name)
                    .with_context(|| format!("adapter artifact is missing tensor {name}"))?;
                tensors.insert(
                    format!("{PEFT_ROOT}.{module}.{factor}.weight"),
                    tensor.contiguous()?,
                );
            }
            let leaf = module.rsplit('.').next().unwrap_or(module).to_owned();
            if !modules.contains(&leaf) {
                modules.push(leaf);
            }
        }
    }

    fs::create_dir_all(output).with_context(|| format!("failed to create {}", output.display()))?;
    let weights = output.join("adapter_model.safetensors");
    candle_core::safetensors::save(&tensors, &weights)
        .with_context(|| format!("failed to write {}", weights.display()))?;
    let model_revision = source
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
    fs::write(&config_path, serde_json::to_vec_pretty(&adapter_config)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    // A served adapter is loaded beside the base's tokenizer; shipping the
    // base's own copy keeps the pair from drifting to another tokenizer.
    let tokenizer = output.join("tokenizer.json");
    fs::copy(&source.tokenizer, &tokenizer).with_context(|| {
        format!(
            "failed to copy {} to {}",
            source.tokenizer.display(),
            tokenizer.display()
        )
    })?;
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
