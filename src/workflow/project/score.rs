//! A text scored token by token along a direction (`docs/guide/steering/
//! geometry.md`, "Scoring a text token by token"): every token's state at one
//! layer projected onto the artifact's unit direction there.

use anyhow::Result;
use serde::Serialize;

use crate::failures::{refuse, workflow::project as causes};
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

/// Refuses a text with no character but whitespace; `source` says where it
/// came from (`--input`, `--input-file notes.txt`, `the request's text`).
/// It reads no model, so a command checks it before loading one.
pub fn check_text(text: &str, source: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(refuse(
            &causes::TEXT_BLANK,
            &[("source", source), ("bytes", &text.len().to_string())],
        ));
    }
    Ok(())
}

/// `text` read through `runtime` and each token's state at `layer`
/// projected onto `artifact`'s unit direction at that layer.
pub fn score(
    runtime: &Runtime,
    artifact: &SteeringArtifact,
    text: &str,
    layer: usize,
) -> Result<ScoreReport> {
    check_text(text, "the text given")?;
    runtime.check_steering(artifact)?;
    let carried: Vec<String> = artifact
        .vectors
        .iter()
        .map(|vector| vector.layer.to_string())
        .collect();
    let carried = carried.join(", ");
    let layer_shown = layer.to_string();
    let at = [
        ("trait", artifact.trait_name.as_str()),
        ("layer", layer_shown.as_str()),
        ("carried", carried.as_str()),
    ];
    let Some(vector) = artifact.vectors.iter().find(|vector| vector.layer == layer) else {
        return Err(refuse(&causes::LAYER_UNCARRIED, &at));
    };
    let direction: Vec<f64> = vector
        .values
        .iter()
        .map(|&value| f64::from(value))
        .collect();
    let length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !length.is_normal() {
        let (length, width) = (length.to_string(), direction.len().to_string());
        let mut values = at.to_vec();
        values.extend([("length", length.as_str()), ("width", width.as_str())]);
        return Err(refuse(&causes::DIRECTION_LENGTHLESS, &values));
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
            score: state
                .iter()
                .zip(&direction)
                .map(|(s, d)| f64::from(*s) * d)
                .sum::<f64>()
                / length,
        })
        .collect();
    let scores = tokens.iter().map(|token| token.score);
    let (min, max) = scores
        .clone()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s), hi.max(s))
        });
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
