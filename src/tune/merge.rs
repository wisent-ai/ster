//! merge.rs — folding an adapter into the base weights.
//!
//! A LoRA adapter is a pair of factors applied beside a projection at every
//! forward pass. That is the right shape while it is being trained and while it
//! is one of several a caller might swap between, and the wrong shape the
//! moment it is finished and permanent: it costs two extra matmuls per adapted
//! projection per token forever, and it means the model cannot be handed to
//! anything that does not know what a Ster artifact is.
//!
//! Merging computes `W + (alpha / rank) * B @ A` once and writes the result as
//! an ordinary checkpoint directory. The output is deliberately not a Ster
//! format: it is `model.safetensors` beside the source's own `config.json` and
//! `tokenizer.json`, which is precisely what `Runtime::load` accepts and what
//! every other tool in the ecosystem accepts too. A merge that produced
//! something only Ster could read would have converted a portable adapter into
//! an unportable model.
//!
//! Two properties are worth stating:
//!
//! * **No decoder is built.** Merging rewrites tensors; it never runs the
//!   model. It resolves the same files through the same Hub path and the same
//!   architecture refusal, and maps nothing.
//! * **The base dtype survives.** The delta is computed in F32 and cast back to
//!   whatever the source weight was, so a BF16 checkpoint merges to a BF16
//!   checkpoint of the same size. Accumulating in the source dtype instead
//!   would round twice and, at BF16's eight bits of mantissa, would quietly
//!   discard small updates entirely.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use crate::{lora, runtime::Checkpoint, workflow};

#[derive(Debug, Clone, Serialize)]
pub struct MergeReport {
    pub model: String,
    pub model_revision: Option<String>,
    pub adapter: String,
    pub output: String,
    pub rank: usize,
    pub alpha: f64,
    /// The constant the low-rank product was multiplied by, `alpha / rank`.
    pub scale: f64,
    pub targets: Vec<String>,
    pub layers: Vec<usize>,
    pub hidden_size: usize,
    /// Projections that received an update.
    pub merged_tensors: usize,
    /// Tensors copied through untouched — embeddings, norms, the head.
    pub copied_tensors: usize,
    pub total_tensors: usize,
    pub parameters: usize,
    /// The dtype the merged projections were written in, which is the dtype
    /// the source wrote them in.
    pub dtype: String,
    /// Every file the merge wrote, relative to the output directory.
    pub files: Vec<String>,
}

/// Folds the adapter at `adapter` into `model` and writes a checkpoint to
/// `output`.
pub fn merge(
    model: &str,
    revision: Option<&str>,
    adapter: &Path,
    output: &Path,
) -> Result<MergeReport> {
    // The CPU is not a fallback here, it is the only sensible device: nothing
    // is computed that a matmul accelerator would help with at this size, and
    // the result is going straight to disk.
    let device = Device::Cpu;
    let artifact = lora::Artifact::load(adapter, &device)?;
    if artifact.kind != lora::Kind::Adapter {
        // A reward model's adapters shaped a residual stream for a head to
        // read. Baking them into a checkpoint and dropping the head produces a
        // model that generates, trained by an objective that never asked it to.
        bail!(
            "adapter artifact is a {} model, not a generation adapter",
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
    let (config, _) = source.llama_config()?;
    if artifact.hidden_size != config.hidden_size {
        bail!(
            "adapter width {} does not match model width {}",
            artifact.hidden_size,
            config.hidden_size
        );
    }
    if let Some(layer) = artifact.layers.iter().copied().find(|layer| *layer >= config.num_hidden_layers)
    {
        bail!(
            "layer {layer} is outside the model's 0..{} range",
            config.num_hidden_layers.saturating_sub(1)
        );
    }

    workflow::progress(format!(
        "reading {} weight file(s) from {model}",
        source.weights.len()
    ));
    let mut tensors: BTreeMap<String, Tensor> = BTreeMap::new();
    for file in &source.weights {
        let loaded = candle_core::safetensors::load(file, &device)
            .with_context(|| format!("failed to read model weights {}", file.display()))?;
        for (name, tensor) in loaded {
            if tensors.insert(name.clone(), tensor).is_some() {
                // Sharded checkpoints partition the tensors; the same name in
                // two shards means the shards disagree, and silently keeping
                // the last one merges half a model.
                bail!("model weight {name} appears in more than one weight file");
            }
        }
    }

    let scale = artifact.alpha / artifact.rank as f64;
    let mut merged = 0usize;
    let mut dtype: Option<DType> = None;
    for &layer in &artifact.layers {
        for &target in &artifact.targets {
            let name = target.checkpoint_tensor(layer);
            let base = tensors
                .get(&name)
                .with_context(|| format!("checkpoint has no tensor {name} to merge into"))?;
            let (a_name, b_name) = lora::Adapter::tensor_names(layer, target);
            let a = artifact
                .tensors
                .get(&a_name)
                .with_context(|| format!("adapter artifact is missing tensor {a_name}"))?;
            let b = artifact
                .tensors
                .get(&b_name)
                .with_context(|| format!("adapter artifact is missing tensor {b_name}"))?;

            // `b @ a` is the dense update the factorisation exists to avoid
            // materialising during training. Here it is materialised exactly
            // once and then thrown away, which is the entire point of merging.
            let original = base.dtype();
            dtype.get_or_insert(original);
            let delta = (b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)? * scale)?;
            let updated = (base.to_dtype(DType::F32)? + &delta).with_context(|| {
                format!("adapter for layer {layer} {} does not fit {name}", target.name())
            })?;
            tensors.insert(name, updated.to_dtype(original)?);
            merged += 1;
        }
    }
    workflow::progress(format!("merged {merged} projections at scale {scale}"));

    fs::create_dir_all(output)
        .with_context(|| format!("failed to create {}", output.display()))?;
    let weights_path = output.join("model.safetensors");
    let flat: HashMap<String, Tensor> =
        tensors.iter().map(|(name, tensor)| (name.clone(), tensor.clone())).collect();
    let parameters: usize = tensors.values().map(|tensor| tensor.elem_count()).sum();
    let total = tensors.len();
    candle_core::safetensors::save(&flat, &weights_path)
        .with_context(|| format!("failed to write {}", weights_path.display()))?;

    // Copied rather than regenerated. The tokenizer and the config are the
    // source's own, and a merged checkpoint that tokenized differently from the
    // model it was merged from would be a different model wearing its name.
    let mut files = vec!["model.safetensors".to_owned()];
    for (from, leaf) in [(&source.config, "config.json"), (&source.tokenizer, "tokenizer.json")] {
        let to = output.join(leaf);
        fs::copy(from, &to)
            .with_context(|| format!("failed to copy {} to {}", from.display(), to.display()))?;
        files.push(leaf.to_owned());
    }
    workflow::progress(format!("wrote {} to {}", files.join(", "), output.display()));

    Ok(MergeReport {
        model: model.to_owned(),
        model_revision: source.revision.clone().or_else(|| artifact.model_revision.clone()),
        adapter: adapter.display().to_string(),
        output: output.display().to_string(),
        rank: artifact.rank,
        alpha: artifact.alpha,
        scale,
        targets: artifact.targets.iter().map(|target| target.name().to_owned()).collect(),
        layers: artifact.layers.clone(),
        hidden_size: artifact.hidden_size,
        merged_tensors: merged,
        copied_tensors: total - merged,
        total_tensors: total,
        parameters,
        // An artifact naming no targets never validates, so `dtype` is set by
        // the time it is read; the fallback keeps the field a string rather
        // than making the report fallible for a case that cannot happen.
        dtype: dtype.map(|value| format!("{value:?}")).unwrap_or_else(|| "unknown".to_owned()),
        files,
    })
}
