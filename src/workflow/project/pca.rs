//! Principal components of a small set of points in a wide space, found on
//! the points' Gram matrix: a few dozen pair sides against a residual stream
//! thousands wide, so the matrix to decompose is the narrow one.

/// The components found, strongest first, and the total variance they are
/// shares of.
pub(super) struct Components {
    /// Unit directions in the points' own space.
    pub(super) axes: Vec<Vec<f64>>,
    /// The variance each axis carries, in the same order.
    pub(super) variances: Vec<f64>,
    /// The total variance of the centred points.
    pub(super) total: f64,
}

pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Up to `count` principal components of `centred`, whose rows are points
/// already centred on their mean. Fewer come back when the points span
/// fewer directions than asked for.
///
/// Each component is the dominant eigenvector of the Gram matrix with the
/// ones before it removed, found by power iteration started from the row
/// of largest norm and stopped when the Rayleigh quotient stops growing —
/// for a positive semi-definite matrix it can only grow, so a step that
/// does not raise it has reached what floating point can resolve.
pub(super) fn principal(centred: &[Vec<f64>], count: usize) -> Components {
    let mut gram: Vec<Vec<f64>> = centred
        .iter()
        .map(|left| centred.iter().map(|right| dot(left, right)).collect())
        .collect();
    let total: f64 = gram.iter().enumerate().map(|(index, row)| row[index]).sum();
    let mut axes = Vec::with_capacity(count);
    let mut variances = Vec::with_capacity(count);
    for _ in std::iter::repeat_n((), count) {
        let Some(start) = gram
            .iter()
            .enumerate()
            .max_by(|(left, row), (right, other)| row[*left].total_cmp(&other[*right]))
            .map(|(_, row)| row.clone())
        else {
            break;
        };
        let length = dot(&start, &start).sqrt();
        if !length.is_normal() {
            break;
        }
        let mut vector: Vec<f64> = start.iter().map(|value| value / length).collect();
        let mut quotient = rayleigh(&gram, &vector);
        loop {
            let image = apply(&gram, &vector);
            let size = dot(&image, &image).sqrt();
            if !size.is_normal() {
                break;
            }
            let next: Vec<f64> = image.iter().map(|value| value / size).collect();
            let raised = rayleigh(&gram, &next);
            if raised <= quotient {
                break;
            }
            vector = next;
            quotient = raised;
        }
        if !quotient.is_normal() {
            break;
        }
        // The same direction in the points' space: the centred rows weighted
        // by the Gram eigenvector, scaled to unit length.
        let mut weighted = vector
            .iter()
            .zip(centred)
            .map(|(weight, row)| row.iter().map(|part| weight * part).collect::<Vec<f64>>());
        let Some(mut axis) = weighted.next() else {
            break;
        };
        for part in weighted {
            axis.iter_mut().zip(part).for_each(|(value, add)| *value += add);
        }
        let axis_length = dot(&axis, &axis).sqrt();
        if !axis_length.is_normal() {
            break;
        }
        axes.push(axis.iter().map(|value| value / axis_length).collect());
        variances.push(quotient);
        // Remove what this component carries before looking for the next.
        for (row, left) in gram.iter_mut().zip(&vector) {
            row.iter_mut()
                .zip(&vector)
                .for_each(|(value, right)| *value -= quotient * left * right);
        }
    }
    Components { axes, variances, total }
}

fn apply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn rayleigh(matrix: &[Vec<f64>], unit: &[f64]) -> f64 {
    dot(unit, &apply(matrix, unit))
}
