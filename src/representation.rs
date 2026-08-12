use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingMethod {
    Caa,
    Pca,
    Logistic,
}

impl TrainingMethod {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "caa" | "mean-difference" => Ok(Self::Caa),
            "pca" => Ok(Self::Pca),
            "logistic" | "probe" => Ok(Self::Logistic),
            _ => bail!("unknown training method {value:?}; expected caa, pca, or logistic"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Caa => "caa",
            Self::Pca => "pca",
            Self::Logistic => "logistic",
        }
    }
}

pub fn train_direction(
    positive: &[Vec<f32>],
    negative: &[Vec<f32>],
    method: TrainingMethod,
) -> Result<Vec<f32>> {
    validate_examples(positive, negative)?;
    let direction = match method {
        TrainingMethod::Caa => mean_difference(positive, negative),
        TrainingMethod::Pca => pca_direction(positive, negative, 64),
        TrainingMethod::Logistic => logistic_direction(positive, negative, 300, 0.1, 1e-4),
    };
    normalize(direction)
}

pub fn evaluate_direction(
    positive: &[Vec<f32>],
    negative: &[Vec<f32>],
    direction: &[f32],
) -> Result<(f32, f32)> {
    validate_examples(positive, negative)?;
    if direction.len() != positive[0].len() {
        bail!(
            "direction width {} does not match activation width {}",
            direction.len(),
            positive[0].len()
        );
    }
    let mut correct = 0usize;
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

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() || left.is_empty() {
        bail!("cosine similarity requires equal non-empty vectors");
    }
    let denominator = dot(left, left).sqrt() * dot(right, right).sqrt();
    if denominator <= f32::EPSILON {
        bail!("cosine similarity is undefined for a zero vector");
    }
    Ok(dot(left, right) / denominator)
}

fn validate_examples(positive: &[Vec<f32>], negative: &[Vec<f32>]) -> Result<()> {
    if positive.is_empty() || positive.len() != negative.len() {
        bail!("training requires the same non-zero number of positive and negative examples");
    }
    let width = positive[0].len();
    if width == 0 {
        bail!("activation vectors are empty");
    }
    if positive
        .iter()
        .chain(negative)
        .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
    {
        bail!("activation vectors must have one finite, consistent width");
    }
    Ok(())
}

fn mean_difference(positive: &[Vec<f32>], negative: &[Vec<f32>]) -> Vec<f32> {
    let mut direction = vec![0f32; positive[0].len()];
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
        .map(|row| row.iter().zip(&mean).map(|(value, mean)| value - mean).collect())
        .collect();
    let mut vector = normalize_or_basis(mean.clone());
    for _ in 0..iterations {
        let mut next = vec![0f32; vector.len()];
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
    let mut weights = vec![0f32; width];
    let count = (positive.len() + negative.len()) as f32;
    for step in 0..iterations {
        let mut gradient = vec![0f32; width];
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
    let mut mean = vec![0f32; rows[0].len()];
    let scale = 1f32 / rows.len() as f32;
    for row in rows {
        for (mean, value) in mean.iter_mut().zip(row) {
            *mean += value * scale;
        }
    }
    mean
}

fn normalize(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    let norm = dot(&vector, &vector).sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        bail!("training produced a zero or non-finite direction");
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
        let mut basis = vec![0f32; vector.len()];
        basis[0] = 1.0;
        basis
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(left, right)| left * right).sum()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
