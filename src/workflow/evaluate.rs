//! Scoring a steering artifact against pairs: how well each layer's
//! direction orders the two sides it was meant to separate.

use anyhow::{Result, bail};
use serde::Serialize;

use crate::{
    artifact::{PairSet, SteeringArtifact},
    representation::evaluate_direction,
    runtime::Runtime,
};

use super::train::capture_pairs;

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
        let layer = captured
            .get(&vector.layer)
            .expect("requested layer is captured");
        let (accuracy, margin) =
            evaluate_direction(&layer.positive, &layer.negative, &vector.values)?;
        reports.push(LayerEvaluation {
            layer: vector.layer,
            accuracy,
            margin,
        });
    }
    Ok(EvaluationReport {
        model: artifact.model.clone(),
        trait_name: artifact.trait_name.clone(),
        method: artifact.method.clone(),
        pair_count: pairs.pairs.len(),
        layers: reports,
    })
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
