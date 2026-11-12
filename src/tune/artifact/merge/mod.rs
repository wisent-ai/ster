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
//! format: it is `model.safetensors` beside the source's own `config.json`
//! (`consolidated.safetensors` beside `params.json` for a Mistral-format
//! source), `tokenizer.json`, and whichever of `tokenizer_config.json` and
//! `chat_template.jinja` the source published — which is precisely what
//! `Runtime::load` accepts and what every other tool in the ecosystem accepts
//! too. A merge that produced something only Ster could read would have
//! converted a portable adapter into an unportable model, and a merge that
//! dropped the chat template would have converted an instruct model into a
//! base model that still answers to its name.
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

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use super::fit::{Fit, Source};
use crate::failures::tune::rewrite::{GGUF_UNMERGEABLE, REWARD_UNMERGEABLE};
use crate::{lora, workflow};

mod checkpoint;

pub(crate) use checkpoint::{copy, read_tensors, tensors_unwritable, unwritable, write_checkpoint};

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
    let fit = Fit {
        step: "merging",
        path: adapter,
        model,
    };
    // A reward model's adapters shaped a residual stream for a head to
    // read. Baking them into a checkpoint and dropping the head produces a
    // model that generates, trained by an objective that never asked it to.
    fit.adapter(&artifact, &REWARD_UNMERGEABLE)?;
    fit.trained_for(&artifact.model)?;
    // A merge writes the base's own tensors back with the delta folded in; a
    // GGUF base holds quantized tensors, and writing them dequantized would
    // publish a model of another size and precision under the base's name.
    let source = fit.source(revision, &GGUF_UNMERGEABLE)?;
    let Source {
        checkpoint: base,
        config,
        architecture,
    } = &source;
    architecture.check_targets(&artifact.targets, config, model)?;
    fit.width(artifact.hidden_size, config.hidden_size)?;
    fit.layers(&artifact.layers, config.num_hidden_layers)?;
    artifact.validate_widths(lora::Widths::for_decoder(config, architecture))?;

    let mut tensors = read_tensors(base, model, &device)?;

    let scale = artifact.alpha / artifact.rank as f64;
    let mut merged = 0usize;
    let mut dtype: Option<DType> = None;
    for &layer in &artifact.layers {
        for &target in &artifact.targets {
            // A fused checkpoint (Phi-3, GLM, GPT-NeoX) has no tensor per
            // projection; the update lands on the rows that projection owns.
            let placement = fit.placement(&source, target, layer)?;
            // A checkpoint saved from the base model (GPT-2's `h.0…` rather
            // than `transformer.h.0…`) carries every name without the root;
            // Mistral's own format stores every name its own way.
            let stored = base.layout.stored_name(&placement.tensor);
            let name = match placement.without_root() {
                Some(rootless) if !tensors.contains_key(&stored) => {
                    base.layout.stored_name(&rootless)
                }
                _ => stored,
            };
            let weight = fit.tensor(&tensors, "the checkpoint", &name, target, layer)?;
            let (a_name, b_name) = lora::Adapter::tensor_names(layer, target);
            let a = fit.tensor(&artifact.tensors, "the adapter", &a_name, target, layer)?;
            let b = fit.tensor(&artifact.tensors, "the adapter", &b_name, target, layer)?;

            // `b @ a` is the dense update the factorisation exists to avoid
            // materialising during training. Here it is materialised exactly
            // once and then thrown away, which is the entire point of merging.
            let original = weight.dtype();
            dtype.get_or_insert(original);
            let updated = fold(weight, a, b, scale, &placement.blocks, placement.transposed)
                .and_then(|updated| updated.to_dtype(original))
                .map_err(|error| fit.unfit(target, layer, &name, weight.dims(), error))?;
            tensors.insert(name, updated);
            merged += 1;
        }
    }
    workflow::progress(format!("merged {merged} projections at scale {scale}"));
    // A projection merged above set `dtype`; an artifact naming no target or
    // no layer merges none and is refused as `Artifact::load` refuses it.
    let Some(dtype) = dtype else {
        let targets: Vec<&str> = artifact.targets.iter().map(|target| target.name()).collect();
        let layers: Vec<String> = artifact.layers.iter().map(usize::to_string).collect();
        return Err(if targets.is_empty() {
            crate::failures::refuse(
                &crate::failures::lora::artifact::TARGETS_NONE,
                &[("layers", &layers.join(","))],
            )
        } else {
            crate::failures::refuse(
                &crate::failures::lora::artifact::LAYERS_NONE,
                &[("targets", &targets.join(","))],
            )
        });
    };

    let written = write_checkpoint(base, output, &tensors)?;
    let (files, parameters, total) = (written.files, written.parameters, written.total);

    Ok(MergeReport {
        model: model.to_owned(),
        model_revision: base
            .revision
            .clone()
            .or_else(|| artifact.model_revision.clone()),
        adapter: adapter.display().to_string(),
        output: output.display().to_string(),
        rank: artifact.rank,
        alpha: artifact.alpha,
        scale,
        targets: artifact
            .targets
            .iter()
            .map(|target| target.name().to_owned())
            .collect(),
        layers: artifact.layers.clone(),
        hidden_size: artifact.hidden_size,
        merged_tensors: merged,
        copied_tensors: total - merged,
        total_tensors: total,
        parameters,
        dtype: format!("{dtype:?}"),
        files,
    })
}

/// `weight` with `scale · b @ a` added, in F32, on the rows `blocks` names
/// or the whole tensor; GPT-2's `Conv1D` stores `[inputs, outputs]` while
/// the update is laid out `[outputs, inputs]`, so a `transposed` weight is
/// turned to match and back.
fn fold(
    weight: &Tensor,
    a: &Tensor,
    b: &Tensor,
    scale: f64,
    blocks: &Option<Vec<(usize, usize, usize)>>,
    transposed: bool,
) -> candle_core::Result<Tensor> {
    let delta = (b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)? * scale)?;
    let widened = weight.to_dtype(DType::F32)?;
    let widened = if transposed { widened.t()? } else { widened };
    let updated = match blocks {
        None => (widened + &delta)?,
        Some(blocks) => blocks.iter().try_fold(widened, |sum, (row, from, rows)| {
            add_rows(&sum, *row, &delta.narrow(0, *from, *rows)?)
        })?,
    };
    if transposed {
        updated.t()?.contiguous()
    } else {
        Ok(updated)
    }
}

/// `base` with `delta` added to its rows `offset..offset + delta.rows`.
fn add_rows(base: &Tensor, offset: usize, delta: &Tensor) -> candle_core::Result<Tensor> {
    let rows = delta.dim(0)?;
    let total = base.dim(0)?;
    let middle = (base.narrow(0, offset, rows)? + delta)?;
    let mut parts = Vec::with_capacity(3);
    if offset > 0 {
        parts.push(base.narrow(0, 0, offset)?);
    }
    parts.push(middle);
    if offset + rows < total {
        parts.push(base.narrow(0, offset + rows, total - offset - rows)?);
    }
    Tensor::cat(&parts, 0)
}
