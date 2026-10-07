//! Reading the two sides of every pair out of the model once, and turning
//! what was read into a steering artifact.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::{
    artifact::{LayerVector, PairSet, SteeringArtifact},
    representation::{TrainingMethod, evaluate_direction, train_direction},
    runtime::Runtime,
};

use super::progress;

pub fn train(
    runtime: &Runtime,
    pairs: &PairSet,
    layers: &[usize],
    method: TrainingMethod,
) -> Result<SteeringArtifact> {
    let captured = capture_pairs(runtime, pairs, layers)?;
    artifact_from_captured(runtime, pairs, &captured, method, layers)
}

#[derive(Debug, Clone)]
pub(super) struct CapturedLayer {
    pub(super) positive: Vec<Vec<f32>>,
    pub(super) negative: Vec<Vec<f32>>,
}

pub(super) type CapturedPairs = BTreeMap<usize, CapturedLayer>;

pub(super) fn capture_pairs(
    runtime: &Runtime,
    pairs: &PairSet,
    layers: &[usize],
) -> Result<CapturedPairs> {
    let mut captured: CapturedPairs = layers
        .iter()
        .map(|layer| {
            (
                *layer,
                CapturedLayer {
                    positive: Vec::with_capacity(pairs.pairs.len()),
                    negative: Vec::with_capacity(pairs.pairs.len()),
                },
            )
        })
        .collect();
    for (index, pair) in pairs.pairs.iter().enumerate() {
        progress(format!("reading pair {}/{}", index + 1, pairs.pairs.len()));
        let positive = runtime.activations(&pair.positive, layers)?;
        let negative = runtime.activations(&pair.negative, layers)?;
        for (layer, values) in positive {
            captured
                .get_mut(&layer)
                .expect("requested layer is captured")
                .positive
                .push(values);
        }
        for (layer, values) in negative {
            captured
                .get_mut(&layer)
                .expect("requested layer is captured")
                .negative
                .push(values);
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
        let layer = captured
            .get(&layer_index)
            .expect("requested layer is captured");
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
