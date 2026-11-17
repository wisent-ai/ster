//! The direction estimators `--method` names, and the seed the one that draws
//! its direction rather than fitting it takes: `random`, the control a fitted
//! direction is compared against, a direction no pair informed.

use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Deserialize;

use crate::failures::{refuse, workflow::direction as causes};

/// A request body names it as the same word `--method` takes; an unknown one
/// is refused while the body is read, with the sentence `parse` writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum TrainingMethod {
    Caa,
    Pca,
    Logistic,
    /// A seeded direction drawn uniformly over the sphere, read from no pair.
    Random,
}

impl TrainingMethod {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "caa" => Ok(Self::Caa),
            "pca" => Ok(Self::Pca),
            "logistic" => Ok(Self::Logistic),
            "random" => Ok(Self::Random),
            _ => Err(refuse(&causes::METHOD_UNKNOWN, &[("value", value)])),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Caa => "caa",
            Self::Pca => "pca",
            Self::Logistic => "logistic",
            Self::Random => "random",
        }
    }
}

impl TryFrom<String> for TrainingMethod {
    type Error = anyhow::Error;
    fn try_from(value: String) -> Result<Self> {
        Self::parse(&value)
    }
}

/// Refuses a `random` direction asked for without a seed, and a seed given to
/// a method that fits its direction from the pairs and would ignore it. A
/// command checks it before any model is loaded.
pub fn check_seed(method: TrainingMethod, seed: Option<u64>) -> Result<()> {
    match (method, seed) {
        (TrainingMethod::Random, None) => Err(refuse(&causes::SEED_MISSING, &[])),
        (TrainingMethod::Random, Some(_)) | (_, None) => Ok(()),
        (fitted, Some(seed)) => Err(refuse(
            &causes::SEED_UNUSED,
            &[("method", fitted.name()), ("seed", &seed.to_string())],
        )),
    }
}

/// The `random` direction at `layer`, `width` values wide: standard normal
/// draws, so once normalized like every fitted direction it points uniformly
/// over the sphere. One stream per seed is read layer after layer, so each
/// layer has its own direction, and one seed and one layer give the same
/// direction whichever other layers a run names.
pub(super) fn random_direction(width: usize, seed: Option<u64>, layer: usize) -> Result<Vec<f32>> {
    let Some(seed) = seed else {
        return Err(refuse(&causes::SEED_MISSING, &[]));
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let draw = || <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng);
    let values: Vec<f32> = std::iter::repeat_with(draw)
        .skip(layer * width)
        .take(width)
        .collect();
    Ok(values)
}
