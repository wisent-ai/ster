use std::{collections::BTreeMap, fs, path::Path};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    artifact::{LayerVector, PairSet, SteeringArtifact},
    representation::{TrainingMethod, evaluate_direction, train_direction},
    runtime::Runtime,
};
/// Progress lines the workflows print while running. The CLI leaves the sink
/// unset and every line goes to stderr; the serve backend installs a sink for
/// the duration of a streamed job so the same lines reach the desktop app as
/// NDJSON log events. Serve runs jobs one at a time, so one global sink is
/// enough.
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

/// The summary document the train/optimize commands print on stdout.
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
            "layers": artifact.vectors.iter().map(|vector| serde_json::json!({
                "layer": vector.layer,
                "train_accuracy": vector.train_accuracy,
                "train_margin": vector.train_margin,
            })).collect::<Vec<_>>()
        }
    })
}

pub fn train(
    runtime: &Runtime,
    pairs: &PairSet,
    layers: &[usize],
    method: TrainingMethod,
) -> Result<SteeringArtifact> {
    let captured = capture_pairs(runtime, pairs, layers)?;
    artifact_from_captured(runtime, pairs, &captured, method, layers)
}

pub fn evaluate(
    runtime: &Runtime,
    pairs: &PairSet,
    artifact: &SteeringArtifact,
) -> Result<EvaluationReport> {
    artifact.validate()?;
    if artifact.model != runtime.model_id {
        bail!(
            "artifact model {:?} does not match runtime model {:?}",
            artifact.model,
            runtime.model_id
        );
    }
    let layers: Vec<usize> = artifact.vectors.iter().map(|vector| vector.layer).collect();
    let captured = capture_pairs(runtime, pairs, &layers)?;
    let mut reports = Vec::with_capacity(artifact.vectors.len());
    for vector in &artifact.vectors {
        let layer = captured.get(&vector.layer).expect("requested layer is captured");
        let (accuracy, margin) = evaluate_direction(&layer.positive, &layer.negative, &vector.values)?;
        reports.push(LayerEvaluation { layer: vector.layer, accuracy, margin });
    }
    Ok(EvaluationReport {
        model: artifact.model.clone(),
        trait_name: artifact.trait_name.clone(),
        method: artifact.method.clone(),
        pair_count: pairs.pairs.len(),
        layers: reports,
    })
}

pub fn optimize(runtime: &Runtime, pairs: &PairSet, layers: &[usize]) -> Result<SteeringArtifact> {
    if pairs.pairs.len() < 4 {
        bail!("optimization requires at least four contrastive pairs");
    }
    let captured = capture_pairs(runtime, pairs, layers)?;
    let split = (pairs.pairs.len() * 4 / 5).clamp(1, pairs.pairs.len() - 1);
    let methods = [TrainingMethod::Caa, TrainingMethod::Pca, TrainingMethod::Logistic];
    let mut best: Option<(f32, f32, usize, TrainingMethod, Vec<f32>)> = None;
    for &layer_index in layers {
        let layer = captured.get(&layer_index).expect("requested layer is captured");
        for method in methods {
            let direction = train_direction(&layer.positive[..split], &layer.negative[..split], method)?;
            let (accuracy, margin) = evaluate_direction(
                &layer.positive[split..],
                &layer.negative[split..],
                &direction,
            )?;
            let candidate = (accuracy, margin, layer_index, method, direction);
            if best.as_ref().is_none_or(|current| {
                candidate.0 > current.0 || (candidate.0 == current.0 && candidate.1 > current.1)
            }) {
                best = Some(candidate);
            }
        }
    }
    let (_, _, layer_index, method, _) = best.expect("methods and layers are non-empty");
    let selected = captured.get(&layer_index).expect("selected layer is captured");
    let direction = train_direction(&selected.positive, &selected.negative, method)?;
    let (accuracy, margin) = evaluate_direction(&selected.positive, &selected.negative, &direction)?;
    let mut artifact = SteeringArtifact::new(
        runtime.model_id.clone(),
        runtime.revision.clone(),
        pairs.trait_name.clone(),
        method.name().to_owned(),
        runtime.hidden_size(),
        vec![LayerVector {
            layer: layer_index,
            values: direction,
            train_margin: margin,
            train_accuracy: accuracy,
        }],
    );
    artifact.metadata.insert("selection".to_owned(), "80/20 holdout over method and layer".to_owned());
    Ok(artifact)
}

pub fn extract(runtime: &Runtime, input: &Path, output: &Path, layers: &[usize]) -> Result<()> {
    let bytes = fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;
    let prompts: PromptSet = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid prompt JSON in {}", input.display()))?;
    if prompts.prompts.is_empty() {
        bail!("prompt set contains no prompts");
    }
    let mut records = Vec::with_capacity(prompts.prompts.len());
    for (index, prompt) in prompts.prompts.iter().enumerate() {
        progress(format!("extracting prompt {}/{}", index + 1, prompts.prompts.len()));
        let activations = runtime.activations(prompt, layers)?;
        records.push(ActivationRecord {
            prompt: prompt.clone(),
            layers: activations.into_iter().collect(),
        });
    }
    let artifact = ActivationArtifact {
        schema_version: 1,
        product: "ster".to_owned(),
        model: runtime.model_id.clone(),
        model_revision: runtime.revision.clone(),
        hidden_size: runtime.hidden_size(),
        records,
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec(&artifact)?)
        .with_context(|| format!("failed to write {}", output.display()))
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

#[derive(Debug, Clone)]
struct CapturedLayer {
    positive: Vec<Vec<f32>>,
    negative: Vec<Vec<f32>>,
}

type CapturedPairs = BTreeMap<usize, CapturedLayer>;

fn capture_pairs(runtime: &Runtime, pairs: &PairSet, layers: &[usize]) -> Result<CapturedPairs> {
    let mut captured: CapturedPairs = layers.iter().map(|layer| {
        (*layer, CapturedLayer { positive: Vec::with_capacity(pairs.pairs.len()), negative: Vec::with_capacity(pairs.pairs.len()) })
    }).collect();
    for (index, pair) in pairs.pairs.iter().enumerate() {
        progress(format!("reading pair {}/{}", index + 1, pairs.pairs.len()));
        let positive = runtime.activations(&pair.positive, layers)?;
        let negative = runtime.activations(&pair.negative, layers)?;
        for (layer, values) in positive {
            captured.get_mut(&layer).expect("requested layer is captured").positive.push(values);
        }
        for (layer, values) in negative {
            captured.get_mut(&layer).expect("requested layer is captured").negative.push(values);
        }
    }
    Ok(captured)
}

fn artifact_from_captured(
    runtime: &Runtime,
    pairs: &PairSet,
    captured: &CapturedPairs,
    method: TrainingMethod,
    layers: &[usize],
) -> Result<SteeringArtifact> {
    let mut vectors = Vec::with_capacity(layers.len());
    for &layer_index in layers {
        let layer = captured.get(&layer_index).expect("requested layer is captured");
        let direction = train_direction(&layer.positive, &layer.negative, method)?;
        let (accuracy, margin) = evaluate_direction(&layer.positive, &layer.negative, &direction)?;
        vectors.push(LayerVector {
            layer: layer_index,
            values: direction,
            train_margin: margin,
            train_accuracy: accuracy,
        });
    }
    Ok(SteeringArtifact::new(
        runtime.model_id.clone(),
        runtime.revision.clone(),
        pairs.trait_name.clone(),
        method.name().to_owned(),
        runtime.hidden_size(),
        vectors,
    ))
}

#[derive(Debug, Serialize)]
pub struct EvaluationReport {
    pub model: String,
    pub trait_name: String,
    pub method: String,
    pub pair_count: usize,
    pub layers: Vec<LayerEvaluation>,
}

#[derive(Debug, Serialize)]
pub struct LayerEvaluation {
    pub layer: usize,
    pub accuracy: f32,
    pub margin: f32,
}

#[derive(Debug, Deserialize)]
struct PromptSet {
    prompts: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ActivationArtifact {
    schema_version: u32,
    product: String,
    model: String,
    model_revision: Option<String>,
    hidden_size: usize,
    records: Vec<ActivationRecord>,
}

#[derive(Debug, Serialize)]
struct ActivationRecord {
    prompt: String,
    layers: BTreeMap<usize, Vec<f32>>,
}
