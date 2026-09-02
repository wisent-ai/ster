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

/// What `optimize` chose, and the evidence it chose on.
///
/// A chooser that publishes only its choice is asking to be trusted. The
/// scores every candidate earned on the holdout are the whole content of the
/// decision, and they cost nothing to carry: they were computed to make it.
pub struct Selection {
    pub artifact: SteeringArtifact,
    pub holdout: Holdout,
    pub candidates: Vec<Candidate>,
}

impl Selection {
    /// The artifact summary every steering command prints, plus the table.
    pub fn summary(&self) -> serde_json::Value {
        let mut summary = artifact_summary(&self.artifact);
        summary
            .as_object_mut()
            .expect("artifact_summary builds an object")
            .insert(
                "selection".to_owned(),
                serde_json::json!({
                    "holdout": self.holdout,
                    "candidates": self.candidates,
                }),
            );
        summary
    }
}

/// One layer-and-method candidate, scored on pairs it was not fitted on.
#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub layer: usize,
    pub method: String,
    pub holdout_accuracy: f32,
    pub holdout_margin: f32,
    /// True for exactly one row: the candidate this run picked.
    pub selected: bool,
}

/// How the pair set was cut.
///
/// Reported rather than assumed, because "80/20" is a ratio and what an
/// operator needs is the two counts it produced. A four-pair set yields a
/// one-pair holdout, and a holdout of one pair is a coin flip dressed as a
/// measurement — which is a fact about the input, not a defect, so it is
/// stated rather than refused.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Holdout {
    pub fit_pairs: usize,
    pub holdout_pairs: usize,
}

pub fn optimize(runtime: &Runtime, pairs: &PairSet, layers: &[usize]) -> Result<Selection> {
    if pairs.pairs.len() < 4 {
        bail!("optimization requires at least four contrastive pairs");
    }
    let captured = capture_pairs(runtime, pairs, layers)?;
    let split = (pairs.pairs.len() * 4 / 5).clamp(1, pairs.pairs.len() - 1);
    let holdout = Holdout { fit_pairs: split, holdout_pairs: pairs.pairs.len() - split };
    progress(format!(
        "fitting each candidate on {} pairs and ranking on a {}-pair holdout",
        holdout.fit_pairs, holdout.holdout_pairs
    ));
    if holdout.holdout_pairs == 1 {
        progress(
            "a one-pair holdout scores every candidate 0 or 1, so this ranking separates almost nothing; add pairs to make the choice mean something".to_owned(),
        );
    }
    let methods = [TrainingMethod::Caa, TrainingMethod::Pca, TrainingMethod::Logistic];
    let mut candidates = Vec::with_capacity(layers.len() * methods.len());
    let mut best: Option<(f32, f32, usize, TrainingMethod)> = None;
    for &layer_index in layers {
        let layer = captured.get(&layer_index).expect("requested layer is captured");
        for method in methods {
            let direction = train_direction(&layer.positive[..split], &layer.negative[..split], method)?;
            let (accuracy, margin) = evaluate_direction(
                &layer.positive[split..],
                &layer.negative[split..],
                &direction,
            )?;
            candidates.push(Candidate {
                layer: layer_index,
                method: method.name().to_owned(),
                holdout_accuracy: accuracy,
                holdout_margin: margin,
                selected: false,
            });
            if best.as_ref().is_none_or(|current| {
                accuracy > current.0 || (accuracy == current.0 && margin > current.1)
            }) {
                best = Some((accuracy, margin, layer_index, method));
            }
        }
    }
    let (_, _, layer_index, method) = best.expect("methods and layers are non-empty");
    // The winner is marked in place rather than moved to the front: the table
    // stays in the order the search walked it, so two runs over the same
    // layers are diffable line for line.
    for candidate in &mut candidates {
        candidate.selected = candidate.layer == layer_index && candidate.method == method.name();
    }
    // The published direction is refitted on every pair, holdout included: the
    // split existed to rank candidates, and once the ranking is done, throwing
    // away a fifth of the evidence would be paying for the measurement twice.
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
        runtime.precision(),
        runtime.chat_status(),
    );
    artifact.metadata.insert(
        "selection".to_owned(),
        format!(
            "chosen over {} candidates on a {}-pair holdout, then refitted on all {} pairs",
            candidates.len(),
            holdout.holdout_pairs,
            pairs.pairs.len()
        ),
    );
    Ok(Selection { artifact, holdout, candidates })
}

pub fn extract(runtime: &Runtime, input: &Path, output: &Path, layers: &[usize]) -> Result<()> {
    let prompts = PromptSet::load(input)?;
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
        runtime.precision(),
        runtime.chat_status(),
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

/// A bare list of prompts: `{"prompts": ["…"]}`.
///
/// `extract` defined this shape and policy optimization reads the same file,
/// so it is one public type rather than two structs that agree by accident.
/// A prompt set states what to ask; what to do with the answers is the
/// command's business.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptSet {
    pub prompts: Vec<String>,
}

impl PromptSet {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let value: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid prompt JSON in {}", path.display()))?;
        if value.prompts.is_empty() {
            bail!("prompt set contains no prompts");
        }
        Ok(value)
    }
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
