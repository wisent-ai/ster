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

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use crate::{
    artifact::SteeringArtifact,
    failures::{
        refuse,
        tune::{
            ablate::{ABLATION_STRENGTH, DIRECTION_EMPTY, WRITE_WIDTH},
            rewrite::{FUSED_UNABLATABLE, GGUF_UNABLATABLE},
        },
    },
    lora, workflow,
};

use super::fit::Fit;
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
    let shown = artifact_path.display().to_string();
    if !strength.is_normal() {
        return Err(refuse(
            &ABLATION_STRENGTH,
            &[
                ("path", &shown),
                ("model", model),
                ("strength", &strength.to_string()),
            ],
        ));
    }
    // Nothing is computed that an accelerator would help with at this size,
    // and the result goes straight to disk.
    let device = Device::Cpu;
    let artifact = SteeringArtifact::load(artifact_path)?;
    let fit = Fit {
        step: "ablating",
        path: artifact_path,
        model,
    };
    fit.trained_for(&artifact.model)?;
    let source = fit.source(revision, &GGUF_UNABLATABLE)?;
    let config = &source.config;
    fit.width(artifact.hidden_size, config.hidden_size)?;
    let layers: Vec<usize> = artifact.vectors.iter().map(|vector| vector.layer).collect();
    fit.layers(&layers, config.num_hidden_layers)?;
    let base = &source.checkpoint;
    let mut tensors = read_tensors(base, model, &device)?;
    let mut rewritten = Vec::new();
    for vector in &artifact.vectors {
        let layer = vector.layer;
        let length = vector
            .values
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        if !length.is_normal() {
            return Err(refuse(
                &DIRECTION_EMPTY,
                &[
                    ("path", &shown),
                    ("layer", &layer.to_string()),
                    ("length", &length.to_string()),
                ],
            ));
        }
        let unit: Vec<f64> = vector
            .values
            .iter()
            .map(|&value| f64::from(value) / length)
            .collect();
        for target in [lora::Target::Output, lora::Target::Down] {
            let placement = fit.placement(&source, target, layer)?;
            fit.whole(&placement, (target, layer), &FUSED_UNABLATABLE, None)?;
            let stored = base.layout.stored_name(&placement.tensor);
            let name = match placement.without_root() {
                Some(rootless) if !tensors.contains_key(&stored) => {
                    base.layout.stored_name(&rootless)
                }
                _ => stored,
            };
            let weight = fit.tensor(&tensors, "the checkpoint", &name, target, layer)?;
            let unfit = |error| fit.unfit(target, layer, &name, weight.dims(), error);
            let original = weight.dtype();
            let widened = weight.to_dtype(DType::F32).map_err(unfit)?;
            // `[outputs, inputs]`, outputs being the residual stream; GPT-2's
            // `Conv1D` stores the other way round and is turned to match.
            let write = if placement.transposed {
                widened.t().map_err(unfit)?
            } else {
                widened
            };
            let rows: Vec<Vec<f32>> = write.to_vec2::<f32>().map_err(unfit)?;
            if rows.len() != unit.len() {
                return Err(refuse(
                    &WRITE_WIDTH,
                    &[
                        ("path", &shown),
                        ("layer", &layer.to_string()),
                        ("name", &name),
                        ("outputs", &rows.len().to_string()),
                        ("width", &unit.len().to_string()),
                    ],
                ));
            }
            let updated = remove_direction(&rows, &unit, strength);
            // A projection of the residual stream has as many rows as the
            // stream is wide, checked above, and the stream is never empty.
            let inputs = updated.first().map_or(0, Vec::len);
            let flat: Vec<f32> = updated.into_iter().flatten().collect();
            let updated = Tensor::from_vec(flat, (unit.len(), inputs), &device)
                .and_then(|updated| {
                    if placement.transposed {
                        updated.t()?.contiguous()
                    } else {
                        Ok(updated)
                    }
                })
                .and_then(|updated| updated.to_dtype(original))
                .map_err(unfit)?;
            tensors.insert(name.clone(), updated);
            rewritten.push(name);
        }
    }
    workflow::progress(format!(
        "rewrote {} residual-stream writes at strength {strength}",
        rewritten.len()
    ));
    let written = write_checkpoint(base, output, &tensors)?;
    Ok(AblationReport {
        model: model.to_owned(),
        model_revision: base
            .revision
            .clone()
            .or_else(|| artifact.model_revision.clone()),
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
        row.iter()
            .map(|&value| weight * f64::from(value))
            .collect::<Vec<f64>>()
    });
    let Some(first) = weighted.next() else {
        return Vec::new();
    };
    let along = weighted.fold(first, |mut sum, part| {
        sum.iter_mut()
            .zip(part)
            .for_each(|(total, add)| *total += add);
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
