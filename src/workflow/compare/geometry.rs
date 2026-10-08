//! The vector arithmetic a comparison of steering directions needs, in
//! `f64` so a sum over a residual stream's width keeps its low bits.

use anyhow::{Result, bail};

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// A direction scaled to unit length. One without a usable length has no
/// angle to anything; it is refused by name rather than reported as
/// orthogonal.
pub(super) fn unit(vector: &[f64], label: &str, layer: usize) -> Result<Vec<f64>> {
    let length = dot(vector, vector).sqrt();
    if !length.is_normal() {
        bail!("{label}'s direction at layer {layer} has length {length}, so it has no angle to compare");
    }
    Ok(vector.iter().map(|value| value / length).collect())
}

/// Distance between two unit directions, taking whichever sign of `right`
/// lies nearer: a steering direction and its negation pick out one axis.
pub(super) fn chord(left: &[f64], right: &[f64]) -> f64 {
    let apart: f64 = left.iter().zip(right).map(|(a, b)| (a - b) * (a - b)).sum();
    let opposed: f64 = left.iter().zip(right).map(|(a, b)| (a + b) * (a + b)).sum();
    apart.min(opposed).sqrt()
}

/// What remains of `vector` after removing its projection on the span of
/// `others`, orthonormalized one by one. A direction already inside the
/// span of the ones before it adds nothing and is passed over.
pub(super) fn outside_span(vector: &[f64], others: &[&Vec<f64>]) -> Vec<f64> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(others.len());
    for other in others {
        let rest = remove_projection(other, &basis);
        let length = dot(&rest, &rest).sqrt();
        if length.is_normal() {
            basis.push(rest.iter().map(|value| value / length).collect());
        }
    }
    remove_projection(vector, &basis)
}

fn remove_projection(vector: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut rest = vector.to_vec();
    for axis in basis {
        let along = dot(&rest, axis);
        rest.iter_mut().zip(axis).for_each(|(value, unit)| *value -= along * unit);
    }
    rest
}

/// The element-wise mean of equally sized matrices, shaped like the first.
pub(super) fn mean(matrices: &[&Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    let count = matrices.len() as f64;
    let Some(first) = matrices.first() else {
        return Vec::new();
    };
    first
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(column, _)| matrices.iter().map(|matrix| matrix[row][column]).sum::<f64>() / count)
                .collect()
        })
        .collect()
}
