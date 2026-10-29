//! Create an untrained local checkpoint from explicit caller-owned inputs.
//! The CLI and native Desktop request share this operation and its diagnostics.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;

use crate::{DeviceChoice, Runtime};

mod spec;
mod weights;

#[derive(Debug, Serialize)]
pub struct ActivationReport {
    pub layer: usize,
    pub width: usize,
}

#[derive(Debug, Serialize)]
pub struct FixtureReport {
    pub path: PathBuf,
    pub configuration: Value,
    pub declaration: Value,
    pub tensor_count: usize,
    pub verification: Vec<ActivationReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_verification: Option<crate::tokenizer::VerificationReport>,
}

/// Create a new checkpoint directory. Existing output is never overwritten.
///
/// `declaration` names a JSON document whose config and tokenizer paths resolve
/// relative to that document. Initialization and verification inputs are
/// required. Success means Ster loaded the emitted F32 CPU checkpoint and
/// actually forwarded the declared prompt through every requested layer.
/// Failed writes or verification remove only this invocation's output files;
/// failed cleanup is reported alongside the original operation's cause.
pub fn create(declaration: &Path, output: &Path) -> Result<FixtureReport> {
    let output_text = output.to_str().with_context(|| {
        format!(
            "fixture output {} cannot be passed to Ster's UTF-8 checkpoint loader",
            output.display()
        )
    })?;
    ensure!(
        !output
            .try_exists()
            .with_context(|| format!("inspect fixture output {}", output.display()))?,
        "fixture output {} already exists; choose a new directory; existing files were not changed",
        output.display()
    );
    let input = spec::Resolved::read(declaration)?;
    let tensors = weights::build(&input)?;
    let tensor_count = tensors.len();
    let configuration = serde_json::to_value(&input.dimensions)
        .context("serialize the declared fixture configuration")?;
    let declaration = serde_json::to_value(&input.spec)
        .context("serialize the accepted fixture declaration and its source annotations")?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create fixture output parent {}", parent.display()))?;
    }
    fs::create_dir(output)
        .with_context(|| format!("create new fixture output {}", output.display()))?;
    let mut owned_files = Vec::new();
    let result = (|| -> Result<FixtureReport> {
        // Native checkpoint filenames, retained as external wire identities:
        // https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained
        let config = output.join("config.json");
        owned_files.push(config.clone());
        fs::write(&config, &input.config_bytes)
            .with_context(|| format!("write fixture config {}", config.display()))?;
        let tokenizer = output.join("tokenizer.json");
        owned_files.push(tokenizer.clone());
        fs::write(&tokenizer, &input.tokenizer_bytes)
            .with_context(|| format!("write fixture tokenizer {}", tokenizer.display()))?;
        let checkpoint = output.join("model.safetensors");
        owned_files.push(checkpoint.clone());
        candle_core::safetensors::save(&tensors, &checkpoint).with_context(|| {
            format!(
                "serialize native fixture tensors to {}",
                checkpoint.display()
            )
        })?;
        // The real consumer maps the saved checkpoint; release the producer's
        // tensor buffers before loading it instead of retaining both copies.
        drop(tensors);
        let runtime = Runtime::load(output_text, None, DeviceChoice::Cpu)
            .with_context(|| format!("load emitted fixture {} through Ster", output.display()))?;
        let activations = runtime
            .activations(
                &input.spec.verification.prompt,
                &input.spec.verification.layers,
            )
            .with_context(|| {
                format!(
                    "forward fixture verification prompt through {} at layers {:?}",
                    output.display(),
                    input.spec.verification.layers
                )
            })?;
        let mut verification = Vec::with_capacity(activations.len());
        for (layer, values) in activations {
            ensure!(
                values.len() == runtime.hidden_size(),
                "fixture verification layer {layer} returned width {}; Ster loaded hidden size {}",
                values.len(),
                runtime.hidden_size()
            );
            for (index, value) in values.iter().enumerate() {
                ensure!(
                    value.is_finite(),
                    "fixture verification layer {layer} returned non-finite activation {value} at element {index}"
                );
            }
            verification.push(ActivationReport {
                layer,
                width: values.len(),
            });
        }
        Ok(FixtureReport {
            path: output.to_owned(),
            configuration,
            declaration,
            tensor_count,
            verification,
            tokenizer_verification: input.tokenizer_verification,
        })
    })();
    match result {
        Ok(report) => Ok(report),
        Err(error) => {
            let mut failures = Vec::new();
            for path in owned_files.into_iter().rev() {
                match fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(cause) if cause.kind() == ErrorKind::NotFound => {}
                    Err(cause) => {
                        failures.push(format!("remove fixture file {}: {cause}", path.display()))
                    }
                }
            }
            if let Err(cause) = fs::remove_dir(output) {
                failures.push(format!(
                    "remove fixture output directory {}: {cause}",
                    output.display()
                ));
            }
            if failures.is_empty() {
                Err(error.context(format!(
                    "fixture creation failed; this invocation's output {} was removed",
                    output.display()
                )))
            } else {
                Err(error.context(format!(
                    "fixture creation failed and output cleanup also failed: {}",
                    failures.join("; ")
                )))
            }
        }
    }
}
