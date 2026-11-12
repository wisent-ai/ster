//! The layers a comparison runs at: the ones named, each carried by every
//! artifact, or every layer all of them carry.

use std::collections::BTreeSet;

use anyhow::Result;

use crate::artifact::SteeringArtifact;
use crate::failures::{refuse, workflow::compare as causes};

/// The layers asked for, each carried by every artifact, or every layer all
/// of them carry when none were named.
pub(super) fn compared_layers(
    artifacts: &[SteeringArtifact],
    labels: &[String],
    named: Option<&[usize]>,
) -> Result<Vec<usize>> {
    let carried: Vec<BTreeSet<usize>> = artifacts
        .iter()
        .map(|artifact| artifact.vectors.iter().map(|vector| vector.layer).collect())
        .collect();
    let common: BTreeSet<usize> = carried
        .iter()
        .skip(1)
        .fold(carried[0].clone(), |common, layers| {
            common.intersection(layers).copied().collect()
        });
    let shown = |layers: &BTreeSet<usize>| {
        let listed: Vec<String> = layers.iter().map(usize::to_string).collect();
        if listed.is_empty() { "none".to_owned() } else { listed.join(", ") }
    };
    // An empty list (a request's `"layers": []`) names no layer, so every
    // shared layer is compared, as when none is named.
    if let Some(named) = named.filter(|named| !named.is_empty()) {
        for &layer in named {
            for (artifact, layers) in artifacts.iter().zip(&carried) {
                if !layers.contains(&layer) {
                    let asked: Vec<String> = named.iter().map(usize::to_string).collect();
                    return Err(refuse(
                        &causes::LAYER_UNCARRIED,
                        &[
                            ("labels", &labels.join(", ")),
                            ("layer", &layer.to_string()),
                            ("label", &artifact.trait_name),
                            ("carried", &shown(layers)),
                            ("named", &asked.join(",")),
                            ("common", &shown(&common)),
                        ],
                    ));
                }
            }
        }
        return Ok(named.to_vec());
    }
    if common.is_empty() {
        let listed: Vec<String> = artifacts
            .iter()
            .zip(&carried)
            .map(|(artifact, layers)| format!("{} [{}]", artifact.trait_name, shown(layers)))
            .collect();
        return Err(refuse(
            &causes::LAYERS_UNSHARED,
            &[("labels", &labels.join(", ")), ("listed", &listed.join("; "))],
        ));
    }
    Ok(common.into_iter().collect())
}

