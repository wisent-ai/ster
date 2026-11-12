use std::path::Path;

use crate::failures::file;
use crate::{artifact::Document, lora};

/// What a training record is, for the `ster.file` refusals that name it.
const FORM: &str = "the JSON ster writes beside an adapter, or a steering artifact";

/// The object recording how an artifact was produced. A sidecar read or JSON
/// failure is observable, as its declared `ster.file` refusal; missing
/// metadata and binary adapter weights are not evidence that the training
/// configuration matched the current runtime.
pub(super) fn read(artifact: &Path) -> anyhow::Result<Option<serde_json::Value>> {
    let sidecar = lora::Artifact::sidecar_path(artifact);
    let (bytes, from_sidecar) = match std::fs::read(&sidecar) {
        Ok(bytes) => (bytes, true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::read(artifact) {
                Ok(bytes) => (bytes, false),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(file::read::refusal("the artifact's training record", artifact, &error)),
            }
        }
        Err(error) => return Err(file::read::refusal("the adapter's sidecar", &sidecar, &error)),
    };
    let document = match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(document) => document,
        Err(error) if from_sidecar => {
            return Err(file::json::refusal("the adapter's sidecar", FORM, &sidecar, &error));
        }
        // Without a sidecar the artifact itself is read: bytes that are not
        // JSON at all are adapter weights, which hold no record. Any other
        // failure is said, naming the artifact.
        Err(error) => match error.classify() {
            serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
                return Ok(None);
            }
            serde_json::error::Category::Io | serde_json::error::Category::Data => {
                return Err(file::json::refusal("the artifact's training record", FORM, artifact, &error));
            }
        },
    };
    // Steering documents already use the JSON extension, so their sidecar
    // path is the artifact itself. The document shape, not its path, decides
    // whether the record is at the top level or inside an adapter's train.
    Ok(Some(match Document::recognise(&bytes) {
        Document::Steering => document,
        Document::AdapterSidecar | Document::Unrecognised => document["train"].clone(),
    }))
}
