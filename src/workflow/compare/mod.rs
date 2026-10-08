//! Comparing steering vectors fitted for different traits on one model:
//! how alike their directions are layer by layer, how much of each one no
//! other direction explains, and which of them group together. The only
//! choice is the caller's: how many groups to cut the artifacts into.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use anyhow::{Result, bail};
use serde::Serialize;

use crate::artifact::SteeringArtifact;

mod geometry;
mod linkage;

use geometry::{chord, dot, outside_span, unit};

/// What the caller decides about one comparison.
pub struct CompareOptions {
    /// Layers to compare; `None` compares every layer all artifacts carry.
    pub layers: Option<Vec<usize>>,
    /// How many groups the artifacts are cut into, at most their count.
    pub clusters: NonZeroUsize,
}

#[derive(Debug, Serialize)]
pub struct Pair {
    pub a: String,
    pub b: String,
    pub cosine: f64,
}

#[derive(Debug, Serialize)]
pub struct LayerSimilarity {
    pub layer: usize,
    /// Cosine similarity of every pair of directions at this layer, in
    /// label order.
    pub similarity: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct Uniqueness {
    pub label: String,
    /// Length of the part of this artifact's directions that the other
    /// artifacts' directions at the same layers do not span, over their
    /// whole length, across the compared layers.
    pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct ComparisonReport {
    pub labels: Vec<String>,
    pub layers: Vec<usize>,
    /// Mean over the compared layers of each pair's cosine similarity.
    pub similarity: Vec<Vec<f64>>,
    pub per_layer: Vec<LayerSimilarity>,
    /// Mean over the compared layers of the distance between each pair's
    /// unit directions, whichever of their two signs is nearer: the
    /// distance the groups are cut on.
    pub distance: Vec<Vec<f64>>,
    pub uniqueness: Vec<Uniqueness>,
    /// Group of each artifact, in label order, from average linkage on
    /// `distance`.
    pub clusters: Vec<usize>,
    /// Mean silhouette of that grouping on the same distances, over the
    /// artifacts that share their group with another; none when every
    /// artifact is in one group or alone in its own.
    pub silhouette: Option<f64>,
    pub most_similar: Pair,
    pub most_different: Pair,
}

/// Compare `artifacts`, labelled by their trait names. They must come from
/// one model at one width: a direction is a point in one model's residual
/// stream, and two models' streams share no basis to compare in.
pub fn compare(artifacts: &[SteeringArtifact], options: &CompareOptions) -> Result<ComparisonReport> {
    let [first, _second, ..] = artifacts else {
        bail!(
            "vector compare needs at least two artifacts and got {}",
            artifacts.len()
        );
    };
    for artifact in artifacts {
        if artifact.model != first.model || artifact.hidden_size != first.hidden_size {
            bail!(
                "vector compare compares directions of one model: {} is {} at width {}, {} is {} at width {}",
                first.trait_name,
                first.model,
                first.hidden_size,
                artifact.trait_name,
                artifact.model,
                artifact.hidden_size
            );
        }
    }
    if options.clusters.get() > artifacts.len() {
        bail!(
            "--clusters {} cannot group {} artifacts: at most one group per artifact",
            options.clusters,
            artifacts.len()
        );
    }
    let labels: Vec<String> = artifacts.iter().map(|artifact| artifact.trait_name.clone()).collect();
    let layers = compared_layers(artifacts, options.layers.as_deref())?;

    let stacks: Vec<Vec<Vec<f64>>> = layers
        .iter()
        .map(|&layer| artifacts.iter().map(|artifact| direction(artifact, layer)).collect())
        .collect();
    let mut per_layer = Vec::with_capacity(layers.len());
    let mut chords = Vec::with_capacity(layers.len());
    for (&layer, stack) in layers.iter().zip(&stacks) {
        let mut units = Vec::with_capacity(stack.len());
        for (vector, label) in stack.iter().zip(&labels) {
            units.push(unit(vector, label, layer)?);
        }
        let similarity = units
            .iter()
            .map(|left| units.iter().map(|right| dot(left, right)).collect())
            .collect();
        let apart: Vec<Vec<f64>> = units
            .iter()
            .map(|left| units.iter().map(|right| chord(left, right)).collect())
            .collect();
        chords.push(apart);
        per_layer.push(LayerSimilarity { layer, similarity });
    }
    let similarity = geometry::mean(&per_layer.iter().map(|layer| &layer.similarity).collect::<Vec<_>>());
    let distance = geometry::mean(&chords.iter().collect::<Vec<_>>());

    let uniqueness = labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            let mut residual = Vec::with_capacity(stacks.len());
            let mut whole = Vec::with_capacity(stacks.len());
            for stack in &stacks {
                let vector = &stack[index];
                let others: Vec<&Vec<f64>> = stack
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(_, other)| other)
                    .collect();
                let rest = outside_span(vector, &others);
                residual.push(dot(&rest, &rest));
                whole.push(dot(vector, vector));
            }
            Uniqueness {
                label: label.clone(),
                score: (residual.iter().sum::<f64>() / whole.iter().sum::<f64>()).sqrt(),
            }
        })
        .collect();

    let clusters = linkage::average(&distance, options.clusters.get());
    let silhouette = linkage::silhouette(&distance, &clusters);
    let (most_similar, most_different) = extremes(&labels, &similarity);

    Ok(ComparisonReport {
        labels,
        layers,
        similarity,
        per_layer,
        distance,
        uniqueness,
        clusters,
        silhouette,
        most_similar,
        most_different,
    })
}

/// The layers asked for, each carried by every artifact, or every layer all
/// of them carry when none were named.
fn compared_layers(artifacts: &[SteeringArtifact], named: Option<&[usize]>) -> Result<Vec<usize>> {
    let carried: Vec<BTreeSet<usize>> = artifacts
        .iter()
        .map(|artifact| artifact.vectors.iter().map(|vector| vector.layer).collect())
        .collect();
    if let Some(named) = named {
        for &layer in named {
            for (artifact, layers) in artifacts.iter().zip(&carried) {
                if !layers.contains(&layer) {
                    bail!(
                        "layer {layer} is not in {}'s artifact, which carries layers {:?}",
                        artifact.trait_name,
                        layers
                    );
                }
            }
        }
        return Ok(named.to_vec());
    }
    let common = carried.iter().fold(None::<BTreeSet<usize>>, |common, layers| {
        Some(match common {
            None => layers.clone(),
            Some(common) => common.intersection(layers).copied().collect(),
        })
    });
    match common {
        Some(common) if !common.is_empty() => Ok(common.into_iter().collect()),
        _ => {
            let listed: Vec<String> = artifacts
                .iter()
                .zip(&carried)
                .map(|(artifact, layers)| format!("{} {:?}", artifact.trait_name, layers))
                .collect();
            bail!("the artifacts share no layer to compare at: {}", listed.join(", "))
        }
    }
}

fn direction(artifact: &SteeringArtifact, layer: usize) -> Vec<f64> {
    artifact
        .vectors
        .iter()
        .find(|vector| vector.layer == layer)
        .map(|vector| vector.values.iter().map(|&value| f64::from(value)).collect())
        .expect("compared_layers keeps only layers every artifact carries")
}

fn extremes(labels: &[String], similarity: &[Vec<f64>]) -> (Pair, Pair) {
    let pairs: Vec<(usize, usize, f64)> = similarity
        .iter()
        .enumerate()
        .flat_map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .filter(move |(column, _)| *column > row)
                .map(move |(column, &cosine)| (row, column, cosine))
        })
        .collect();
    let pair = |&(row, column, cosine): &(usize, usize, f64)| Pair {
        a: labels[row].clone(),
        b: labels[column].clone(),
        cosine,
    };
    let most_similar = pairs
        .iter()
        .max_by(|left, right| left.2.total_cmp(&right.2))
        .map(pair)
        .expect("two or more artifacts make at least one pair");
    let most_different = pairs
        .iter()
        .min_by(|left, right| left.2.total_cmp(&right.2))
        .map(pair)
        .expect("two or more artifacts make at least one pair");
    (most_similar, most_different)
}
