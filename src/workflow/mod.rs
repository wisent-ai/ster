//! The four things Ster does with a model, each in its own module: reading
//! pairs into a steering artifact (`train`), scoring one (`evaluate`),
//! choosing a layer and method on a holdout (`optimize`), and exporting raw
//! representations (`extract`). This entry keeps only what all four share —
//! the progress channel, the artifact summary every command prints, and the
//! layer-selection parser.

use std::sync::Mutex;

use anyhow::{bail, Context, Result};

use crate::artifact::SteeringArtifact;

mod evaluate;
mod extract;
mod optimize;
mod train;

pub use evaluate::{evaluate, EvaluationReport, LayerEvaluation};
pub use extract::{extract, PromptSet};
pub use optimize::{optimize, Candidate, Holdout, Selection};
pub use train::train;

/// Progress lines the workflows print while running. The CLI leaves the sink
/// unset and every line goes to stderr; `ster request` installs a sink so the
/// same lines reach the desktop app as NDJSON log events. One process runs
/// one request, so one global sink is enough.
static PROGRESS_SINK: Mutex<Option<Box<dyn Fn(&str) + Send>>> = Mutex::new(None);

pub fn set_progress_sink(sink: Option<Box<dyn Fn(&str) + Send>>) {
    *PROGRESS_SINK.lock().expect("progress sink lock") = sink;
}

pub fn progress(message: String) {
    let guard = PROGRESS_SINK.lock().expect("progress sink lock");
    match guard.as_ref() {
        Some(sink) => sink(&message),
        None => eprintln!("{message}"),
    }
}

/// The document `train`, `optimize` and `inspect` print.
///
/// It describes every vector rather than printing one: `ster inspect` used to
/// serialize the artifact itself, which on a twenty-two-layer checkpoint is
/// forty-five thousand floats down a terminal, while `ster tune inspect`
/// printed shapes. A steering vector's content is not readable and its shape
/// and length are, so this reports what a reader can actually use — and the
/// artifact is still on disk for anything that wants the numbers.
pub fn artifact_summary(artifact: &SteeringArtifact) -> serde_json::Value {
    serde_json::json!({
        "artifact": {
            "schema_version": artifact.schema_version,
            "product": artifact.product,
            "model": artifact.model,
            "model_revision": artifact.model_revision,
            "trait_name": artifact.trait_name,
            "method": artifact.method,
            "hidden_size": artifact.hidden_size,
            "precision": artifact.precision,
            "chat_template": artifact.chat_template,
            "layers": artifact.vectors.iter().map(|vector| serde_json::json!({
                "layer": vector.layer,
                "width": vector.values.len(),
                "norm": norm(&vector.values),
                "train_accuracy": vector.train_accuracy,
                "train_margin": vector.train_margin,
            })).collect::<Vec<_>>(),
            "metadata": artifact.metadata,
        }
    })
}

/// The Euclidean length of one direction, accumulated in `f64` so a
/// two-thousand-term sum does not lose its low bits.
fn norm(values: &[f32]) -> f64 {
    values.iter().map(|value| f64::from(*value) * f64::from(*value)).sum::<f64>().sqrt()
}


pub fn parse_layers(value: &str, count: usize) -> Result<Vec<usize>> {
    if count == 0 {
        bail!("model has no layers");
    }
    if value == "all" {
        return Ok((0..count).collect());
    }
    let mut layers = Vec::new();
    for segment in value.split(',').map(str::trim).filter(|segment| !segment.is_empty()) {
        if let Some((start, end)) = segment.split_once("..") {
            let start: usize = start.parse().with_context(|| format!("invalid layer range {segment:?}"))?;
            let end: usize = end.parse().with_context(|| format!("invalid layer range {segment:?}"))?;
            if start >= end {
                bail!("layer range {segment:?} must have start < end");
            }
            layers.extend(start..end);
        } else {
            layers.push(segment.parse().with_context(|| format!("invalid layer {segment:?}"))?);
        }
    }
    layers.sort_unstable();
    layers.dedup();
    if layers.is_empty() {
        bail!("no layers selected");
    }
    if let Some(layer) = layers.iter().copied().find(|layer| *layer >= count) {
        bail!("layer {layer} is outside the model's 0..{} range", count - 1);
    }
    Ok(layers)
}

