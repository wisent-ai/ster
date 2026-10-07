//! Choosing a layer and a method on pairs the candidate was not fitted on,
//! and publishing every score the choice was made from.

use anyhow::{Result, bail};
use serde::Serialize;

use crate::{
    artifact::{LayerVector, PairSet, SteeringArtifact},
    representation::{TrainingMethod, evaluate_direction, train_direction},
    runtime::Runtime,
};

use super::{artifact_summary, progress, train::capture_pairs};

/// What `optimize` chose, and the evidence it chose on.
///
/// A chooser that publishes only its choice is asking to be trusted. The
/// scores every candidate earned on the holdout are the whole content of the
/// decision, and they cost nothing to carry: they were computed to make it.
pub struct Selection {
    pub artifact: SteeringArtifact,
    pub holdout: Holdout,
    pub candidates: Vec<Candidate>,
}

impl Selection {
    /// The artifact summary every steering command prints, plus the table.
    pub fn summary(&self) -> serde_json::Value {
        let mut summary = artifact_summary(&self.artifact);
        summary
            .as_object_mut()
            .expect("artifact_summary builds an object")
            .insert(
                "selection".to_owned(),
                serde_json::json!({
                    "holdout": self.holdout,
                    "candidates": self.candidates,
                }),
            );
        summary
    }
}

/// One layer-and-method candidate, scored on pairs it was not fitted on.
#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub layer: usize,
    pub method: String,
    pub holdout_accuracy: f32,
    pub holdout_margin: f32,
    /// True for exactly one row: the candidate this run picked.
    pub selected: bool,
}

/// How the pair set was cut.
///
/// Reported rather than assumed: what an operator needs is the two counts
/// the stated fraction produced. A holdout of one pair is a coin flip
/// dressed as a measurement — a fact about the input, not a defect, so it is
/// stated rather than refused.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Holdout {
    pub fraction: f64,
    pub fit_pairs: usize,
    pub holdout_pairs: usize,
}

/// Fits every candidate on the first part of `pairs` and ranks it on the
/// held-out `holdout` fraction, rounded to whole pairs. Ster assumes no
/// fraction; one that leaves either side without a pair is refused.
pub fn optimize(
    runtime: &Runtime,
    pairs: &PairSet,
    layers: &[usize],
    holdout: f64,
) -> Result<Selection> {
    if !(holdout > 0.0 && holdout < 1.0) {
        bail!("the held-out fraction must be above zero and below one");
    }
    let total = pairs.pairs.len();
    let holdout_pairs = (total as f64 * holdout).round() as usize;
    if holdout_pairs == 0 || holdout_pairs >= total {
        bail!(
            "a held-out fraction of {holdout} over {total} contrastive pairs leaves {} pairs to fit and {holdout_pairs} to rank on; both need at least one",
            total.saturating_sub(holdout_pairs)
        );
    }
    let captured = capture_pairs(runtime, pairs, layers)?;
    let split = total - holdout_pairs;
    let holdout = Holdout {
        fraction: holdout,
        fit_pairs: split,
        holdout_pairs,
    };
    progress(format!(
        "fitting each candidate on {} pairs and ranking on a {}-pair holdout",
        holdout.fit_pairs, holdout.holdout_pairs
    ));
    if holdout.holdout_pairs == 1 {
        progress(
            "a one-pair holdout scores every candidate 0 or 1, so this ranking separates almost nothing; add pairs to make the choice mean something".to_owned(),
        );
    }
    let methods = [
        TrainingMethod::Caa,
        TrainingMethod::Pca,
        TrainingMethod::Logistic,
    ];
    let mut candidates = Vec::with_capacity(layers.len() * methods.len());
    let mut best: Option<(f32, f32, usize, TrainingMethod)> = None;
    for &layer_index in layers {
        let layer = captured
            .get(&layer_index)
            .expect("requested layer is captured");
        for method in methods {
            let direction =
                train_direction(&layer.positive[..split], &layer.negative[..split], method)?;
            let (accuracy, margin) = evaluate_direction(
                &layer.positive[split..],
                &layer.negative[split..],
                &direction,
            )?;
            candidates.push(Candidate {
                layer: layer_index,
                method: method.name().to_owned(),
                holdout_accuracy: accuracy,
                holdout_margin: margin,
                selected: false,
            });
            if best.as_ref().is_none_or(|current| {
                accuracy > current.0 || (accuracy == current.0 && margin > current.1)
            }) {
                best = Some((accuracy, margin, layer_index, method));
            }
        }
    }
    let (_, _, layer_index, method) = best.expect("methods and layers are non-empty");
    // The winner is marked in place rather than moved to the front: the table
    // stays in the order the search walked it, so two runs over the same
    // layers are diffable line for line.
    for candidate in &mut candidates {
        candidate.selected = candidate.layer == layer_index && candidate.method == method.name();
    }
    // The published direction is refitted on every pair, holdout included: the
    // split existed to rank candidates, and once the ranking is done, throwing
    // away a fifth of the evidence would be paying for the measurement twice.
    let selected = captured
        .get(&layer_index)
        .expect("selected layer is captured");
    let direction = train_direction(&selected.positive, &selected.negative, method)?;
    let (accuracy, margin) =
        evaluate_direction(&selected.positive, &selected.negative, &direction)?;
    let mut artifact = SteeringArtifact::new(
        runtime.model_id.clone(),
        runtime.revision.clone(),
        pairs.trait_name.clone(),
        method.name().to_owned(),
        runtime.hidden_size(),
        vec![LayerVector {
            layer: layer_index,
            values: direction,
            train_margin: margin,
            train_accuracy: accuracy,
        }],
        runtime.precision(),
        runtime.chat_status(),
    );
    artifact.metadata.insert(
        "selection".to_owned(),
        format!(
            "chosen over {} candidates on a {}-pair holdout, then refitted on all {} pairs",
            candidates.len(),
            holdout.holdout_pairs,
            pairs.pairs.len()
        ),
    );
    Ok(Selection {
        artifact,
        holdout,
        candidates,
    })
}
