//! The plan that adds steering artifacts to a loaded model's residual
//! stream, and the checks every artifact passes before it touches the model.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use anyhow::Result;

use crate::{
    artifact::SteeringArtifact,
    failures::{
        refuse,
        runtime::steer::{MODEL_OTHER, STRENGTH_NOT_FINITE, STRENGTHS_ZERO, WIDTH_OTHER},
    },
    model::SteeringPlan,
};

use super::super::{Runtime, validate_layers};

impl Runtime {
    /// The plan that adds `artifact`'s vectors, scaled by `strength`, to this
    /// model's residual stream — refused when the artifact was fitted on
    /// another model, at another width, or names a layer this model lacks.
    ///
    /// Strength selection and projection steer with one artifact, so both
    /// read it through these checks rather than each keeping its own copy.
    pub fn steering_plan(
        &self,
        artifact: &SteeringArtifact,
        strength: f64,
    ) -> Result<SteeringPlan> {
        self.check_steering(artifact)?;
        SteeringPlan::new(
            artifact
                .vectors
                .iter()
                .map(|vector| (vector.layer, vector.values.clone())),
            strength,
            self.hidden_size(),
            &self.device,
            self.dtype,
        )
    }

    /// One plan adding every artifact in `parts` at its own strength: at each
    /// layer the sum of every part's vector times its strength. The plan's
    /// scale is the strength of largest magnitude and each vector is summed
    /// at its ratio to it, so the product the decoder adds is exactly that
    /// sum. `None` for no parts; refused when every strength is zero or one
    /// is not finite, since such a mix steers nothing or nothing sensible.
    pub fn steering_plan_mixed(
        &self,
        parts: &[(&SteeringArtifact, f64)],
    ) -> Result<Option<SteeringPlan>> {
        let strengths: Vec<f64> = parts.iter().map(|(_, strength)| *strength).collect();
        let named: Vec<String> = strengths.iter().map(f64::to_string).collect();
        let named = named.join(" --strength ");
        if let Some(bad) = strengths.iter().find(|strength| !strength.is_finite()) {
            return Err(refuse(
                &STRENGTH_NOT_FINITE,
                &[("strength", &bad.to_string()), ("strengths", &named)],
            ));
        }
        let Some(scale) = strengths
            .iter()
            .copied()
            .max_by(|left, right| left.abs().total_cmp(&right.abs()))
        else {
            return Ok(None);
        };
        if !scale.is_normal() {
            return Err(refuse(&STRENGTHS_ZERO, &[("strengths", &named)]));
        }
        let mut sums: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for (artifact, strength) in parts {
            self.check_steering(artifact)?;
            let ratio = strength / scale;
            for vector in &artifact.vectors {
                let scaled = vector.values.iter().map(|&value| f64::from(value) * ratio);
                match sums.entry(vector.layer) {
                    Entry::Vacant(slot) => {
                        slot.insert(scaled.collect());
                    }
                    Entry::Occupied(mut slot) => {
                        slot.get_mut()
                            .iter_mut()
                            .zip(scaled)
                            .for_each(|(sum, part)| *sum += part);
                    }
                }
            }
        }
        Ok(Some(SteeringPlan::new(
            sums.into_iter().map(|(layer, values)| {
                (
                    layer,
                    values.into_iter().map(|value| value as f32).collect(),
                )
            }),
            scale,
            self.hidden_size(),
            &self.device,
            self.dtype,
        )?))
    }

    /// The checks every steering artifact passes before it touches this
    /// model: a valid document, fitted for this model, at this width, at
    /// layers this model has.
    pub fn check_steering(&self, artifact: &SteeringArtifact) -> Result<()> {
        artifact.validate()?;
        let model = self.model_id.as_str();
        let trait_name = artifact.trait_name.as_str();
        if artifact.model != model {
            return Err(refuse(
                &MODEL_OTHER,
                &[
                    ("model", model),
                    ("trait_name", trait_name),
                    ("trained", &format!("{:?}", artifact.model)),
                ],
            ));
        }
        if artifact.hidden_size != self.hidden_size() {
            return Err(refuse(
                &WIDTH_OTHER,
                &[
                    ("model", model),
                    ("trait_name", trait_name),
                    ("trained", &artifact.hidden_size.to_string()),
                    ("width", &self.hidden_size().to_string()),
                ],
            ));
        }
        validate_layers(
            &artifact
                .vectors
                .iter()
                .map(|vector| vector.layer)
                .collect::<Vec<_>>(),
            self.layer_count(),
        )
    }
}
