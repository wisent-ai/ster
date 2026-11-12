//! strength.rs — choosing how hard to push a steering artifact, on pairs.
//!
//! A direction says which way to move the residual stream; how far is a
//! separate question the direction cannot answer, and generation refuses to
//! guess it. Each strength the operator names is measured the way BiPO's
//! objective reads a policy: the steered model's log-probability of each side
//! of a pair minus the unsteered model's. A pair is ordered when steering
//! raised its positive side more than its negative side, and its shift is the
//! difference. A strength that pushes too hard drags both sides down together
//! and orders fewer pairs, which is what this measures and a representation
//! score cannot.
//!
//! The choice follows the rule `optimize` ranks layers and methods by: the
//! largest share of pairs ordered, the larger mean shift breaking a tie. Every
//! candidate's numbers are reported, because they are the whole decision.
//! Ster names no candidate strength, no batch and no sequence limit.

use anyhow::Result;
use serde::Serialize;

use super::super::super::{
    batch,
    encode::{encode_pairs, pair_set_label},
};
use super::scoring::{Scored, log_ratios, reference_scores};
use crate::{
    artifact::{PairSet, SteeringArtifact},
    failures::{
        refuse,
        tune::{
            run::score_failed,
            strength::{SETTING_ZERO, STRENGTH_NOT_FINITE, STRENGTHS_NONE},
        },
    },
    runtime::Runtime,
    workflow,
};

/// Every setting one strength selection takes; Ster assumes none of them.
#[derive(Debug, Clone)]
pub struct StrengthOptions {
    /// The scales to measure, in the order they are reported.
    pub strengths: Vec<f64>,
    pub batch: usize,
    pub max_sequence: usize,
}

/// One candidate strength and what it did to the pairs.
#[derive(Debug, Clone, Serialize)]
pub struct StrengthCandidate {
    pub strength: f64,
    /// Share of scored pairs whose positive side steering raised more than
    /// their negative side.
    pub ordered: f64,
    /// Mean over scored pairs of the positive side's log-ratio minus the
    /// negative side's.
    pub mean_shift: f64,
    /// True for exactly one row: the strength this run picked.
    pub selected: bool,
}

/// What a strength selection measured and chose.
#[derive(Debug, Clone, Serialize)]
pub struct StrengthReport {
    pub method: String,
    pub layers: Vec<usize>,
    pub pairs: usize,
    pub scored_pairs: usize,
    pub skipped_long: usize,
    pub selected_strength: f64,
    pub candidates: Vec<StrengthCandidate>,
}

impl StrengthOptions {
    /// Refuses a selection whose settings cannot measure anything, before
    /// any model or pair is read; each refusal names its setting.
    pub fn check(&self) -> Result<()> {
        if self.strengths.is_empty() {
            return Err(refuse(
                &STRENGTHS_NONE,
                &[
                    ("batch", &self.batch.to_string()),
                    ("max_sequence", &self.max_sequence.to_string()),
                ],
            ));
        }
        if let Some(strength) = self.strengths.iter().find(|value| !value.is_finite()) {
            let named: Vec<String> = self.strengths.iter().map(f64::to_string).collect();
            return Err(refuse(
                &STRENGTH_NOT_FINITE,
                &[
                    ("strength", &strength.to_string()),
                    ("strengths", &named.join(",")),
                ],
            ));
        }
        for (name, flag, value) in [
            ("batch size", "batch-size", self.batch),
            ("sequence limit", "max-sequence", self.max_sequence),
        ] {
            if value == 0 {
                return Err(refuse(&SETTING_ZERO, &[("name", name), ("flag", flag)]));
            }
        }
        Ok(())
    }
}

/// Measures every strength in `options` with `artifact` added to `runtime`'s
/// residual stream over `pairs`, and picks one.
pub fn strengths(
    runtime: &Runtime,
    pairs: &PairSet,
    artifact: &SteeringArtifact,
    options: &StrengthOptions,
) -> Result<StrengthReport> {
    options.check()?;
    pairs.validate(&pair_set_label(pairs))?;
    // Past its context the model's positions are untrained, and its batched
    // forward holds no more: the limit is clamped as every trainer's is.
    let limit = options.max_sequence.min(runtime.context_length()?);
    let mut encoded: Vec<Scored> = encode_pairs(runtime, pairs, limit, "measure")?
        .into_iter()
        .map(Scored::new)
        .collect();
    let skipped_long = pairs.pairs.len() - encoded.len();
    reference_scores(runtime, &mut encoded, options.batch)?;

    let mut candidates: Vec<StrengthCandidate> = Vec::with_capacity(options.strengths.len());
    for &strength in &options.strengths {
        let measured = || -> Result<(Vec<bool>, Vec<f64>)> {
            let steering = runtime.steering_plan(artifact, strength)?;
            let mut ordered = Vec::with_capacity(encoded.len());
            let mut shifts = Vec::with_capacity(encoded.len());
            for group in encoded.chunks(options.batch) {
                let mut rows: Vec<&[u32]> = Vec::with_capacity(group.len() + group.len());
                for scored in group {
                    rows.push(&scored.pair.chosen);
                    rows.push(&scored.pair.rejected);
                }
                let sides = rows.len() / group.len();
                let read = batch::read_rows(&rows, options.batch, sides, |pass| {
                    runtime.forward_steered_scored_rows(pass, &steering)
                })?;
                for (scored, logits) in group.iter().zip(read.chunks_exact(sides)) {
                    let [chosen_logits, rejected_logits] = logits else {
                        unreachable!("every pair is read as its two sides");
                    };
                    let (chosen, rejected) =
                        log_ratios(runtime, scored, chosen_logits, rejected_logits)?;
                    ordered.push(chosen > rejected);
                    shifts.push(chosen - rejected);
                }
            }
            Ok((ordered, shifts))
        };
        let (ordered, shifts) = measured()
            .map_err(|error| score_failed(&format!("the pairs at strength {strength}"), error))?;
        let count = shifts.len() as f64;
        let candidate = StrengthCandidate {
            strength,
            ordered: ordered.iter().filter(|&&raised| raised).count() as f64 / count,
            mean_shift: shifts.iter().sum::<f64>() / count,
            selected: false,
        };
        workflow::progress(format!(
            "strength {strength}: {:.3} of pairs ordered, mean shift {:.4}",
            candidate.ordered, candidate.mean_shift
        ));
        candidates.push(candidate);
    }

    let chosen = candidates
        .iter()
        .enumerate()
        .reduce(|best, current| {
            let (_, leader) = best;
            let (_, challenger) = current;
            if challenger.ordered > leader.ordered
                || (challenger.ordered == leader.ordered
                    && challenger.mean_shift > leader.mean_shift)
            {
                current
            } else {
                best
            }
        })
        .map(|(slot, _)| slot)
        .expect("check refuses an empty strength list");
    candidates[chosen].selected = true;
    Ok(StrengthReport {
        method: artifact.method.clone(),
        layers: artifact.vectors.iter().map(|vector| vector.layer).collect(),
        pairs: pairs.pairs.len(),
        scored_pairs: encoded.len(),
        skipped_long,
        selected_strength: candidates[chosen].strength,
        candidates,
    })
}
