//! Verifying a written fixture: the declared prompt forwarded through the
//! checkpoint Ster loaded, every requested layer as wide as the model and
//! every activation finite.

use anyhow::Result;

use crate::Runtime;
use crate::failures::{fixture::output as causes, refuse};

use super::ActivationReport;
use super::spec::Verification;

/// The width of every layer `verification` names, read by forwarding its
/// prompt through `runtime`, the checkpoint at `path`.
pub(super) fn activations(
    runtime: &Runtime,
    verification: &Verification,
    path: &str,
) -> Result<Vec<ActivationReport>> {
    let activations = runtime.activations(&verification.prompt, &verification.layers)?;
    let hidden = runtime.hidden_size();
    let mut reports = Vec::with_capacity(activations.len());
    for (layer, values) in activations {
        if values.len() != hidden {
            return Err(refuse(
                &causes::VERIFICATION_WIDTH,
                &[
                    ("path", path),
                    ("layer", &layer.to_string()),
                    ("width", &values.len().to_string()),
                    ("hidden", &hidden.to_string()),
                ],
            ));
        }
        let unfinite = values.iter().filter(|value| !value.is_finite()).count();
        if let Some((index, value)) = values.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(refuse(
                &causes::VERIFICATION_NOT_FINITE,
                &[
                    ("path", path),
                    ("layer", &layer.to_string()),
                    ("index", &index.to_string()),
                    ("value", &value.to_string()),
                    ("count", &unfinite.to_string()),
                    ("width", &values.len().to_string()),
                ],
            ));
        }
        reports.push(ActivationReport {
            layer,
            width: values.len(),
        });
    }
    Ok(reports)
}
