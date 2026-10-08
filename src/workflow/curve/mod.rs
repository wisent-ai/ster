//! How many pairs a direction needs: one method fitted on growing first
//! parts of a pair set and scored on the same held-out pairs every time, so
//! the score each size earns can be read off against the others. The port
//! of wisent's optimize-sample-size, with every size the caller's.

use std::num::NonZeroUsize;

use anyhow::{Result, bail};
use serde::Serialize;

use crate::{
    artifact::PairSet,
    representation::{TrainingMethod, evaluate_direction, train_direction},
    runtime::Runtime,
};

use super::{
    Holdout,
    optimize::{effect_size, split_holdout},
    progress,
    train::capture_pairs,
};

/// One size at one layer, scored on the held-out pairs.
#[derive(Debug, Clone, Serialize)]
pub struct CurvePoint {
    pub size: usize,
    pub layer: usize,
    pub holdout_accuracy: f32,
    pub holdout_margin: f32,
    /// Cohen's d of the held-out projections, as `vector optimize` reports it.
    pub holdout_effect_size: Option<f64>,
}

/// Per layer, the smallest size that reached the best held-out accuracy any
/// size reached there.
#[derive(Debug, Clone, Serialize)]
pub struct Enough {
    pub layer: usize,
    pub best_accuracy: f32,
    pub smallest_size: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CurveReport {
    pub trait_name: String,
    pub method: String,
    pub holdout: Holdout,
    pub points: Vec<CurvePoint>,
    pub enough: Vec<Enough>,
}

/// Fit `method` at every layer on the first `size` pairs of the fitting
/// part, for every size in `sizes`, and score each on the held-out last
/// `holdout` fraction of the set. Every size must fit inside the fitting
/// part; Ster chooses no size.
pub fn curve(
    runtime: &Runtime,
    pairs: &PairSet,
    layers: &[usize],
    method: TrainingMethod,
    holdout: f64,
    sizes: &[NonZeroUsize],
) -> Result<CurveReport> {
    let holdout = split_holdout(pairs.pairs.len(), holdout)?;
    if sizes.is_empty() {
        bail!("vector curve needs at least one size to fit on");
    }
    if let Some(size) = sizes.iter().find(|size| size.get() > holdout.fit_pairs) {
        bail!(
            "a size of {size} pairs does not fit in the {} pairs left to fit on after holding out {}",
            holdout.fit_pairs,
            holdout.holdout_pairs
        );
    }
    let captured = capture_pairs(runtime, pairs, layers)?;
    let split = holdout.fit_pairs;
    progress(format!(
        "fitting {} on {} sizes at {} layers and scoring each on a {}-pair holdout",
        method.name(),
        sizes.len(),
        layers.len(),
        holdout.holdout_pairs
    ));
    let mut points = Vec::with_capacity(sizes.len() * layers.len());
    for &layer in layers {
        let read = captured.get(&layer).expect("requested layer is captured");
        for size in sizes {
            let size = size.get();
            let direction = train_direction(&read.positive[..size], &read.negative[..size], method)?;
            let (accuracy, margin) =
                evaluate_direction(&read.positive[split..], &read.negative[split..], &direction)?;
            points.push(CurvePoint {
                size,
                layer,
                holdout_accuracy: accuracy,
                holdout_margin: margin,
                holdout_effect_size: effect_size(&read.positive[split..], &read.negative[split..], &direction),
            });
        }
    }
    let enough = layers
        .iter()
        .filter_map(|&layer| {
            let at_layer: Vec<&CurvePoint> = points.iter().filter(|point| point.layer == layer).collect();
            let best = at_layer
                .iter()
                .map(|point| point.holdout_accuracy)
                .max_by(f32::total_cmp)?;
            let smallest = at_layer
                .iter()
                .filter(|point| point.holdout_accuracy == best)
                .map(|point| point.size)
                .min()?;
            Some(Enough {
                layer,
                best_accuracy: best,
                smallest_size: smallest,
            })
        })
        .collect();
    Ok(CurveReport {
        trait_name: pairs.trait_name.clone(),
        method: method.name().to_owned(),
        holdout,
        points,
        enough,
    })
}
