//! ablate.rs — writing a steering direction into the weights themselves.
//!
//! A steering vector is added to the residual stream on every forward pass;
//! a permanent change is one the checkpoint carries with no artifact beside
//! it. At every layer the artifact carries, the two projections that write
//! the residual stream — the attention output and the feed-forward down
//! projection — are rewritten as `W - strength * v (v^T W)` with `v` the
//! layer's unit direction: at strength one the layer can no longer write
//! along the direction at all, a smaller strength removes that share of it,
//! and a negative one amplifies it. The result is an ordinary checkpoint
//! directory every loader reads, written by the same code `tune merge`
//! writes with. This is wisent's modify-weights by directional projection.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use crate::{
    artifact::SteeringArtifact,
    lora,
    runtime::{Checkpoint, Layout},
    workflow,
};

use super::merge::{read_tensors, write_checkpoint};

#[derive(Debug, Clone, Serialize)]
pub struct AblationReport {
    pub model: String,
    pub model_revision: Option<String>,
    pub artifact: String,
    pub trait_name: String,
    pub output: String,
    pub strength: f64,
    pub layers: Vec<usize>,
    /// The residual-stream writes that were rewritten, by stored name.
    pub rewritten: Vec<String>,
    /// Tensors copied through untouched.
    pub copied_tensors: usize,
    pub total_tensors: usize,
    pub parameters: usize,
    /// Every file written, relative to the output directory.
    pub files: Vec<String>,
}

/// Rewrites `model`'s residual-stream writes at every layer `artifact`
/// carries so they remove `strength` of its direction, and writes the
/// checkpoint to `output`.
pub fn ablate(
    model: &str,
    revision: Option<&str>,
    artifact_path: &Path,
    strength: f64,
    output: &Path,
) -> Result<AblationReport> {
    if !strength.is_normal() {
        bail!("ablation strength {strength} changes nothing or nothing sensible; state a finite, non-zero strength");
    }
    // Nothing is computed that an accelerator would help with at this size,
    // and the result goes straight to disk.
    let device = Device::Cpu;
    let artifact = SteeringArtifact::load(artifact_path)?;
    if artifact.model != model {
        bail!(
            "artifact was trained for model {:?}, current model is {:?}",
            artifact.model,
            model
        );
    }
    let source = Checkpoint::resolve(model, revision)?;
    if source.layout == Layout::Gguf {
        bail!("{model} is a GGUF checkpoint; a direction is written into safetensors weights only");
    }
    let (config, architecture, _) = source.decoder_config()?;
    if artifact.hidden_size != config.hidden_size {
        bail!(
            "artifact width {} does not match model width {}",
            artifact.hidden_size,
            config.hidden_size
        );
    }
    if let Some(layer) = artifact
        .vectors
        .iter()
        .map(|vector| vector.layer)
        .find(|layer| *layer >= config.num_hidden_layers)
    {
        bail!(
            "layer {layer} is outside a model of {} layers",
            config.num_hidden_layers
        );
    }
    let mut tensors = read_tensors(&source, model, &device)?;
    let mut rewritten = Vec::new();
    for vector in &artifact.vectors {
        let length = vector
            .values
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        if !length.is_normal() {
            bail!(
                "the direction at layer {} has length {length}, so there is nothing to remove",
                vector.layer
            );
        }
        let unit: Vec<f64> = vector.values.iter().map(|&value| f64::from(value) / length).collect();
        for target in [lora::Target::Output, lora::Target::Down] {
            let placement = architecture
                .placement(target, vector.layer, &config)
                .with_context(|| {
                    format!(
                        "this model has no {} projection at layer {} to rewrite",
                        target.name(),
                        vector.layer
                    )
                })?;
            if placement.blocks.is_some() {
                bail!(
                    "layer {} {} is stored fused with another projection; rewriting one part of a fused write is not supported",
                    vector.layer,
                    target.name()
                );
            }
            let stored = source.layout.stored_name(&placement.tensor);
            let name = match placement.without_root() {
                Some(rootless) if !tensors.contains_key(&stored) => source.layout.stored_name(&rootless),
                _ => stored,
            };
            let base = tensors
                .get(&name)
                .with_context(|| format!("checkpoint has no tensor {name} to rewrite"))?;
            let original = base.dtype();
            let widened = base.to_dtype(DType::F32)?;
            // `[outputs, inputs]`, outputs being the residual stream; GPT-2's
            // `Conv1D` stores the other way round and is turned to match.
            let write = if placement.transposed { widened.t()? } else { widened };
            let rows: Vec<Vec<f32>> = write.to_vec2::<f32>()?;
            if rows.len() != unit.len() {
                bail!(
                    "{name} writes {} outputs where the residual stream is {} wide",
                    rows.len(),
                    unit.len()
                );
            }
            let updated = remove_direction(&rows, &unit, strength);
            let inputs = updated.first().map(Vec::len).context("a projection has rows")?;
            let flat: Vec<f32> = updated.into_iter().flatten().collect();
            let updated = Tensor::from_vec(flat, (unit.len(), inputs), &device)?;
            let updated = if placement.transposed { updated.t()?.contiguous()? } else { updated };
            tensors.insert(name.clone(), updated.to_dtype(original)?);
            rewritten.push(name);
        }
    }
    workflow::progress(format!(
        "rewrote {} residual-stream writes at strength {strength}",
        rewritten.len()
    ));
    let written = write_checkpoint(&source, output, &tensors)?;
    Ok(AblationReport {
        model: model.to_owned(),
        model_revision: source.revision.clone().or_else(|| artifact.model_revision.clone()),
        artifact: artifact_path.display().to_string(),
        trait_name: artifact.trait_name.clone(),
        output: output.display().to_string(),
        strength,
        layers: artifact.vectors.iter().map(|vector| vector.layer).collect(),
        copied_tensors: written.total - rewritten.len(),
        rewritten,
        total_tensors: written.total,
        parameters: written.parameters,
        files: written.files,
    })
}

/// `rows - strength * unit (unit^T rows)`: each column loses `strength` of
/// its component along `unit`, accumulated in `f64`.
fn remove_direction(rows: &[Vec<f32>], unit: &[f64], strength: f64) -> Vec<Vec<f32>> {
    let mut weighted = rows.iter().zip(unit).map(|(row, weight)| {
        row.iter().map(|&value| weight * f64::from(value)).collect::<Vec<f64>>()
    });
    let Some(first) = weighted.next() else {
        return Vec::new();
    };
    let along = weighted.fold(first, |mut sum, part| {
        sum.iter_mut().zip(part).for_each(|(total, add)| *total += add);
        sum
    });
    rows.iter()
        .zip(unit)
        .map(|(row, weight)| {
            row.iter()
                .zip(&along)
                .map(|(&value, part)| (f64::from(value) - strength * weight * part) as f32)
                .collect()
        })
        .collect()
}
