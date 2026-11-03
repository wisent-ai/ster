//! A text scored token by token along a direction (`docs/guide/steering/
//! geometry.md`, "Scoring a text token by token"): every token's state at one
//! layer projected onto the artifact's unit direction there.

use anyhow::{Result, bail};
use serde::Serialize;

use crate::{Runtime, SteeringArtifact};

/// One token and its length along the direction.
#[derive(Debug, Clone, Serialize)]
pub struct TokenScore {
    pub index: usize,
    pub id: u32,
    pub text: String,
    pub score: f64,
}

/// What `ster vector score` answers.
#[derive(Debug, Clone, Serialize)]
pub struct ScoreReport {
    pub model: String,
    pub trait_name: String,
    pub layer: usize,
    pub tokens: Vec<TokenScore>,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

/// `text` read through `runtime` and each token's state at `layer`
/// projected onto `artifact`'s unit direction at that layer.
pub fn score(runtime: &Runtime, artifact: &SteeringArtifact, text: &str, layer: usize) -> Result<ScoreReport> {
    if text.trim().is_empty() {
        bail!("the text is empty; give the text to score");
    }
    runtime.check_steering(artifact)?;
    let Some(vector) = artifact.vectors.iter().find(|vector| vector.layer == layer) else {
        let carried: Vec<String> = artifact.vectors.iter().map(|vector| vector.layer.to_string()).collect();
        bail!(
            "the artifact carries no vector at layer {layer}; it carries layers {}",
            carried.join(", ")
        );
    };
    let direction: Vec<f64> = vector.values.iter().map(|&value| f64::from(value)).collect();
    let length = direction.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !length.is_normal() {
        bail!("the artifact's direction at layer {layer} has no usable length ({length})");
    }
    let read = runtime.token_states(text, layer)?;
    let tokens: Vec<TokenScore> = read
        .states
        .iter()
        .enumerate()
        .map(|(index, state)| TokenScore {
            index,
            id: read.ids[index],
            text: read.texts[index].clone(),
            score: state.iter().zip(&direction).map(|(s, d)| f64::from(*s) * d).sum::<f64>() / length,
        })
        .collect();
    let scores = tokens.iter().map(|token| token.score);
    let (min, max) = scores.clone().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| (lo.min(s), hi.max(s)));
    let mean = scores.sum::<f64>() / tokens.len() as f64;
    Ok(ScoreReport {
        model: artifact.model.clone(),
        trait_name: artifact.trait_name.clone(),
        layer,
        tokens,
        min,
        max,
        mean,
    })
}
