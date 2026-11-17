//! Fitting one layer's steering direction on the reads of the pairs' two
//! sides, or drawing a random one as a control, and scoring a direction by
//! how it orders them. How each fails is `ster.vector.direction`
//! (src/failures/workflow/direction.rs).

use anyhow::Result;
use wisent_errors::Cause;

use crate::failures::{refuse, workflow::direction as causes};

mod estimator;

pub use estimator::{TrainingMethod, check_seed};

/// The unit direction `method` fits at `layer` on the reads of the pairs'
/// positive and negative sides, or, for `random`, draws with `seed` at the
/// pairs' width. Every method's direction is normalized the same way, so a
/// strength means the same on all of them.
pub fn train_direction(
    positive: &[Vec<f32>],
    negative: &[Vec<f32>],
    method: TrainingMethod,
    layer: usize,
    seed: Option<u64>,
) -> Result<Vec<f32>> {
    let step = format!(
        "fitting {} at layer {layer} on {} pairs",
        method.name(),
        positive.len()
    );
    validate_examples(positive, negative, &step)?;
    check_seed(method, seed)?;
    let direction = match method {
        TrainingMethod::Caa => mean_difference(positive, negative),
        TrainingMethod::Pca => pca_direction(positive, negative, 64),
        TrainingMethod::Logistic => logistic_direction(positive, negative, 300, 0.1, 1e-4),
        TrainingMethod::Random => estimator::random_direction(positive[0].len(), seed, layer)?,
    };
    normalize(direction, &step, positive.len())
}

/// How `direction` orders the pairs read at `layer`: the share of pairs
/// whose positive side projects further along it, and the mean margin.
pub fn evaluate_direction(
    positive: &[Vec<f32>],
    negative: &[Vec<f32>],
    direction: &[f32],
    layer: usize,
) -> Result<(f32, f32)> {
    let step = format!(
        "scoring the direction at layer {layer} on {} pairs",
        positive.len()
    );
    validate_examples(positive, negative, &step)?;
    if direction.len() != positive[0].len() {
        return Err(refuse(
            &causes::DIRECTION_WIDTH,
            &[
                ("step", &step),
                ("direction", &direction.len().to_string()),
                ("width", &positive[0].len().to_string()),
            ],
        ));
    }
    let mut correct: usize = Default::default();
    let mut margin = 0f32;
    for (pos, neg) in positive.iter().zip(negative) {
        let pair_margin = dot(pos, direction) - dot(neg, direction);
        if pair_margin > 0.0 {
            correct += 1;
        }
        margin += pair_margin;
    }
    let count = positive.len() as f32;
    Ok((correct as f32 / count, margin / count))
}

/// Refuses reads a direction cannot be fitted or scored on: sides of two
/// counts, an empty first read, reads of two widths, a value not finite.
fn validate_examples(positive: &[Vec<f32>], negative: &[Vec<f32>], step: &str) -> Result<()> {
    let counts = [
        ("step", step.to_owned()),
        ("positive", positive.len().to_string()),
        ("negative", negative.len().to_string()),
    ];
    let refused = |cause: &'static Cause, extra: &[(&str, String)]| {
        let values: Vec<(&str, &str)> = counts
            .iter()
            .chain(extra)
            .map(|(name, value)| (*name, value.as_str()))
            .collect();
        refuse(cause, &values)
    };
    if positive.is_empty() || positive.len() != negative.len() {
        return Err(refused(&causes::EXAMPLES_UNEQUAL, &[]));
    }
    let width = positive[0].len();
    if width == 0 {
        return Err(refused(
            &causes::READ_EMPTY,
            &[("side", "positive".to_owned())],
        ));
    }
    let sides = [("positive", positive), ("negative", negative)];
    for (side, reads) in sides {
        for (index, read) in reads.iter().enumerate() {
            let at = [
                ("side", side.to_owned()),
                ("index", index.to_string()),
                ("width", width.to_string()),
            ];
            if read.len() != width {
                let mut extra = at.to_vec();
                extra.push(("found", read.len().to_string()));
                return Err(refused(&causes::READ_WIDTH, &extra));
            }
            if let Some((position, value)) = read
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                let mut extra = at.to_vec();
                extra.push(("position", position.to_string()));
                extra.push(("value", value.to_string()));
                return Err(refused(&causes::READ_NOT_FINITE, &extra));
            }
        }
    }
    Ok(())
}

fn mean_difference(positive: &[Vec<f32>], negative: &[Vec<f32>]) -> Vec<f32> {
    let mut direction: Vec<f32> = vec![Default::default(); positive[0].len()];
    let scale = 1f32 / positive.len() as f32;
    for (pos, neg) in positive.iter().zip(negative) {
        for ((value, pos), neg) in direction.iter_mut().zip(pos).zip(neg) {
            *value += (pos - neg) * scale;
        }
    }
    direction
}

fn pca_direction(positive: &[Vec<f32>], negative: &[Vec<f32>], iterations: usize) -> Vec<f32> {
    let differences: Vec<Vec<f32>> = positive
        .iter()
        .zip(negative)
        .map(|(pos, neg)| pos.iter().zip(neg).map(|(p, n)| p - n).collect())
        .collect();
    let mean = mean_rows(&differences);
    let centered: Vec<Vec<f32>> = differences
        .iter()
        .map(|row| {
            row.iter()
                .zip(&mean)
                .map(|(value, mean)| value - mean)
                .collect()
        })
        .collect();
    let mut vector = normalize_or_basis(mean.clone());
    for _ in 0..iterations {
        let mut next: Vec<f32> = vec![Default::default(); vector.len()];
        for row in &centered {
            let projection = dot(row, &vector);
            for (value, component) in next.iter_mut().zip(row) {
                *value += projection * component;
            }
        }
        vector = normalize_or_basis(next);
    }
    if dot(&vector, &mean) < 0.0 {
        for value in &mut vector {
            *value = -*value;
        }
    }
    vector
}

fn logistic_direction(
    positive: &[Vec<f32>],
    negative: &[Vec<f32>],
    iterations: usize,
    learning_rate: f32,
    l2: f32,
) -> Vec<f32> {
    let width = positive[0].len();
    let mut weights: Vec<f32> = vec![Default::default(); width];
    let count = (positive.len() + negative.len()) as f32;
    for step in 0..iterations {
        let mut gradient: Vec<f32> = vec![Default::default(); width];
        for (row, label) in positive
            .iter()
            .map(|row| (row, 1f32))
            .chain(negative.iter().map(|row| (row, 0f32)))
        {
            let prediction = sigmoid(dot(row, &weights));
            let error = prediction - label;
            for (grad, value) in gradient.iter_mut().zip(row) {
                *grad += error * value / count;
            }
        }
        let rate = learning_rate / (1.0 + step as f32 * 0.01);
        for (weight, gradient) in weights.iter_mut().zip(gradient) {
            *weight -= rate * (gradient + l2 * *weight);
        }
    }
    weights
}

fn mean_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    let mut mean: Vec<f32> = vec![Default::default(); rows[0].len()];
    let scale = 1f32 / rows.len() as f32;
    for row in rows {
        for (mean, value) in mean.iter_mut().zip(row) {
            *mean += value * scale;
        }
    }
    mean
}

fn normalize(mut vector: Vec<f32>, step: &str, pairs: usize) -> Result<Vec<f32>> {
    let norm = dot(&vector, &vector).sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(refuse(
            &causes::DIRECTION_DEGENERATE,
            &[
                ("step", step),
                ("length", &norm.to_string()),
                ("width", &vector.len().to_string()),
                ("positive", &pairs.to_string()),
            ],
        ));
    }
    for value in &mut vector {
        *value /= norm;
    }
    Ok(vector)
}

fn normalize_or_basis(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = dot(&vector, &vector).sqrt();
    if norm > f32::EPSILON && norm.is_finite() {
        for value in &mut vector {
            *value /= norm;
        }
        vector
    } else {
        let mut basis: Vec<f32> = vec![Default::default(); vector.len()];
        basis[0] = 1.0;
        basis
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
