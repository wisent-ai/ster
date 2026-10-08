//! Where steering moves a model's states: every pair side read at one layer
//! with and without a steering artifact added, projected onto the principal
//! components of the unsteered reads, with the shift measured along the
//! artifact's own direction and toward the positive sides. The picture
//! wisent's steering-viz drew, as the numbers a caller can plot or check.

use std::num::NonZeroUsize;

use anyhow::{Result, bail};
use serde::Serialize;

use crate::{
    artifact::{PairSet, SteeringArtifact},
    runtime::Runtime,
};

use super::progress;

mod pca;

use pca::dot;

/// What the caller decides about one projection.
pub struct ProjectOptions {
    /// The layer every side is read at.
    pub layer: usize,
    /// The strength the artifact is added at for the steered reads.
    pub strength: f64,
    /// How many principal components to project onto.
    pub components: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Positive,
    Negative,
}

#[derive(Debug, Serialize)]
pub struct ProjectedPoint {
    /// Index of the pair in the set.
    pub pair: usize,
    pub side: Side,
    /// Whether this read had the artifact added.
    pub steered: bool,
    /// Coordinates on the components, strongest first.
    pub coordinates: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct Distances {
    pub unsteered: f64,
    pub steered: f64,
}

#[derive(Debug, Serialize)]
pub struct ProjectionReport {
    pub model: String,
    pub trait_name: String,
    pub layer: usize,
    pub strength: f64,
    pub pair_count: usize,
    /// Share of the unsteered reads' variance each component carries; fewer
    /// than asked for when the reads span fewer directions.
    pub explained: Vec<f64>,
    pub points: Vec<ProjectedPoint>,
    /// Mean movement of every read along the artifact's unit direction at
    /// this layer, steered minus unsteered; none when the artifact carries
    /// no vector at this layer.
    pub shift_along_direction: Option<f64>,
    /// Mean distance of the negative sides from the mean of the unsteered
    /// positive sides, before and after steering.
    pub negative_distance_to_positive: Distances,
    /// Share of negative sides that steering brought nearer that mean.
    pub negatives_moved_toward_positive: f64,
}

/// Read every side of `pairs` at `options.layer`, unsteered and with
/// `artifact` added at `options.strength`, and project both onto the
/// principal components of the unsteered reads.
pub fn project(
    runtime: &Runtime,
    pairs: &PairSet,
    artifact: &SteeringArtifact,
    options: &ProjectOptions,
) -> Result<ProjectionReport> {
    let plan = runtime.steering_plan(artifact, options.strength)?;
    let layers = [options.layer];
    let read = |text: &str, steered: bool| -> Result<Vec<f64>> {
        let captured = runtime.activations_steered(text, &layers, steered.then_some(&plan))?;
        let (_, values) = captured
            .into_iter()
            .find(|(layer, _)| *layer == options.layer)
            .ok_or_else(|| anyhow::anyhow!("layer {} was not captured", options.layer))?;
        Ok(values.into_iter().map(f64::from).collect())
    };
    let mut plain = Vec::with_capacity(pairs.pairs.len());
    let mut steered = Vec::with_capacity(pairs.pairs.len());
    for (index, pair) in pairs.pairs.iter().enumerate() {
        progress(format!(
            "reading pair {index} of {}, unsteered and steered",
            pairs.pairs.len()
        ));
        plain.push((read(&pair.positive, false)?, read(&pair.negative, false)?));
        steered.push((read(&pair.positive, true)?, read(&pair.negative, true)?));
    }

    let unsteered_points: Vec<&Vec<f64>> = plain
        .iter()
        .flat_map(|(positive, negative)| [positive, negative])
        .collect();
    let mean = centroid(&unsteered_points);
    let centred: Vec<Vec<f64>> = unsteered_points.iter().map(|point| minus(point, &mean)).collect();
    let fitted = pca::principal(&centred, options.components.get());
    if fitted.axes.is_empty() {
        bail!(
            "the unsteered reads at layer {} do not vary, so they have no component to project onto",
            options.layer
        );
    }
    let coordinates = |point: &[f64]| -> Vec<f64> {
        let offset = minus(point, &mean);
        fitted.axes.iter().map(|axis| dot(&offset, axis)).collect()
    };
    let mut points = Vec::with_capacity(pairs.pairs.len() * fitted.axes.len());
    for (is_steered, reads) in [(false, &plain), (true, &steered)] {
        for (pair, (positive, negative)) in reads.iter().enumerate() {
            for (side, values) in [(Side::Positive, positive), (Side::Negative, negative)] {
                points.push(ProjectedPoint {
                    pair,
                    side,
                    steered: is_steered,
                    coordinates: coordinates(values),
                });
            }
        }
    }

    let direction = artifact
        .vectors
        .iter()
        .find(|vector| vector.layer == options.layer)
        .map(|vector| vector.values.iter().map(|&value| f64::from(value)).collect::<Vec<f64>>());
    let shift_along_direction = direction.and_then(|direction| {
        let length = dot(&direction, &direction).sqrt();
        length.is_normal().then(|| {
            let moves: Vec<f64> = plain
                .iter()
                .zip(&steered)
                .flat_map(|((positive, negative), (steered_positive, steered_negative))| {
                    [minus(steered_positive, positive), minus(steered_negative, negative)]
                })
                .map(|movement| dot(&movement, &direction) / length)
                .collect();
            moves.iter().sum::<f64>() / moves.len() as f64
        })
    });

    let positives: Vec<&Vec<f64>> = plain.iter().map(|(positive, _)| positive).collect();
    let target = centroid(&positives);
    let before: Vec<f64> = plain.iter().map(|(_, negative)| distance(negative, &target)).collect();
    let after: Vec<f64> = steered.iter().map(|(_, negative)| distance(negative, &target)).collect();
    let nearer = before.iter().zip(&after).filter(|(was, now)| now < was).count();

    Ok(ProjectionReport {
        model: artifact.model.clone(),
        trait_name: artifact.trait_name.clone(),
        layer: options.layer,
        strength: options.strength,
        pair_count: pairs.pairs.len(),
        explained: fitted.variances.iter().map(|variance| variance / fitted.total).collect(),
        points,
        shift_along_direction,
        negative_distance_to_positive: Distances {
            unsteered: before.iter().sum::<f64>() / before.len() as f64,
            steered: after.iter().sum::<f64>() / after.len() as f64,
        },
        negatives_moved_toward_positive: nearer as f64 / before.len() as f64,
    })
}

fn minus(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(a, b)| a - b).collect()
}

fn distance(left: &[f64], right: &[f64]) -> f64 {
    let offset = minus(left, right);
    dot(&offset, &offset).sqrt()
}

fn centroid(points: &[&Vec<f64>]) -> Vec<f64> {
    let count = points.len() as f64;
    let mut sums = points.iter().map(|point| point.as_slice());
    let Some(first) = sums.next() else {
        return Vec::new();
    };
    let mut total = first.to_vec();
    for point in sums {
        total.iter_mut().zip(point).for_each(|(value, add)| *value += add);
    }
    total.iter().map(|value| value / count).collect()
}
