//! The things Ster does with a model, each in its own module: reading pairs
//! into a steering artifact (`train`), scoring one (`evaluate`), choosing a
//! layer and method on a holdout (`optimize`), exporting raw
//! representations (`extract`), comparing its forward pass with a
//! reference implementation's (`parity`), and comparing steering artifacts
//! fitted for different traits (`compare`). This entry keeps only what they
//! share — the progress channel, the artifact summary every command prints,
//! and the layer-selection parser.

use std::sync::{Mutex, MutexGuard, PoisonError};

use anyhow::Result;
use wisent_errors::Cause;

use crate::artifact::SteeringArtifact;
use crate::failures::{refuse, workflow::layers as causes};

mod compare;
mod curve;
mod evaluate;
mod extract;
mod optimize;
mod parity;
mod project;
mod train;

pub use compare::{CompareOptions, ComparisonReport, compare};
pub use curve::{CurvePoint, CurveReport, Enough, check_sizes, curve};
pub use evaluate::{EvaluationReport, LayerEvaluation, evaluate};
pub use extract::{PromptSet, extract};
pub use optimize::{Candidate, Holdout, Selection, optimize, split_holdout};
pub use parity::{ParityReport, parity};
pub use project::{
    ProjectOptions, ProjectionReport, ScoreReport, Side, TokenScore, check_text, project, score,
};
pub use train::train;

/// Progress lines the workflows print while running. The CLI leaves the sink
/// unset and every line goes to stderr; `ster request` installs a sink so the
/// same lines reach the desktop app as NDJSON log events. One process runs
/// one request, so one global sink is enough.
static PROGRESS_SINK: Mutex<Sink> = Mutex::new(None);

/// The sink, also after a sink that panicked while it wrote: it holds no
/// state a half-written line could leave wrong.
type Sink = Option<Box<dyn Fn(&str) + Send>>;

fn sink() -> MutexGuard<'static, Sink> {
    PROGRESS_SINK.lock().unwrap_or_else(PoisonError::into_inner)
}

pub fn set_progress_sink(sink_now: Sink) {
    *sink() = sink_now;
}

pub fn progress(message: String) {
    let guard = sink();
    match guard.as_ref() {
        Some(sink) => sink(&message),
        None => eprintln!("{message}"),
    }
}

/// The document `train`, `optimize` and `inspect` print.
///
/// It describes every vector rather than printing one: `ster vector inspect` used to
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
    values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
}

/// `value`, a `--layers` selection, as the layers of a model of `count`
/// layers. `count` is `usize::MAX` where the layers are counted only once
/// the weights are mapped (`ster tune`), so nothing is refused as past them
/// here and `all` is expanded by the caller.
pub fn parse_layers(value: &str, count: usize) -> Result<Vec<usize>> {
    let counted = count != usize::MAX;
    let model_layers = if counted {
        format!(
            "the model has {count} layers, 0 to {}",
            count.saturating_sub(1)
        )
    } else {
        "the model's layers are counted once its weights are mapped".to_owned()
    };
    let example = if counted {
        format!("0..{count}")
    } else {
        "8..16".to_owned()
    };
    let refused = |cause: &'static Cause, extra: &[(&str, &str)]| {
        let mut values = vec![
            ("selection", value),
            ("model_layers", model_layers.as_str()),
            ("example", example.as_str()),
        ];
        values.extend_from_slice(extra);
        refuse(cause, &values)
    };
    if count == 0 {
        return Err(refused(&causes::MODEL_LAYERLESS, &[]));
    }
    if value == "all" {
        return Ok((0..count).collect());
    }
    let mut layers = Vec::new();
    // Every layer named past the model, a range's part as one entry, so the
    // refusal names all of them rather than the first.
    let mut past: Vec<String> = Vec::new();
    let mut first_past = None;
    for segment in value
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        if let Some((start, end)) = segment.split_once("..") {
            let bound = |text: &str| {
                text.trim().parse::<usize>().map_err(|error| {
                    refused(
                        &causes::RANGE_BOUND_UNPARSED,
                        &[
                            ("segment", segment),
                            ("bound", text),
                            ("error", &error.to_string()),
                        ],
                    )
                })
            };
            let (start, end) = (bound(start)?, bound(end)?);
            if start >= end {
                return Err(refused(
                    &causes::RANGE_EMPTY,
                    &[
                        ("segment", segment),
                        ("start", &start.to_string()),
                        ("end", &end.to_string()),
                        ("after", &(start + 1).to_string()),
                    ],
                ));
            }
            if end > count {
                first_past = first_past.or(Some(start.max(count)));
                past.push(format!("{}..{end}", start.max(count)));
            }
            layers.extend(start..end.min(count));
        } else {
            let layer = segment.parse::<usize>().map_err(|error| {
                refused(
                    &causes::LAYER_UNPARSED,
                    &[("segment", segment), ("error", &error.to_string())],
                )
            })?;
            if layer >= count {
                first_past = first_past.or(Some(layer));
                past.push(layer.to_string());
            }
            layers.push(layer);
        }
    }
    if let Some(layer) = first_past {
        return Err(refused(
            &causes::LAYER_PAST,
            &[
                ("layer", &layer.to_string()),
                ("count", &count.to_string()),
                ("last", &(count - 1).to_string()),
                ("past", &past.join(", ")),
            ],
        ));
    }
    layers.sort_unstable();
    layers.dedup();
    if layers.is_empty() {
        return Err(refused(&causes::SELECTION_EMPTY, &[]));
    }
    Ok(layers)
}
