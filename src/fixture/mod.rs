//! Create an untrained local checkpoint from explicit caller-owned inputs.
//! The CLI and native Desktop request share this operation and its diagnostics.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::failures::{file, fixture::output as causes, refuse};
use crate::{DeviceChoice, Runtime};

mod spec;
mod tokenizer;
mod verify;
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
    let shown = output.display().to_string();
    let parent = output.parent().map_or_else(PathBuf::new, Path::to_path_buf);
    let parent_shown = parent.display().to_string();
    let Some(output_text) = output.to_str() else {
        return Err(refuse(&causes::OUTPUT_NOT_UTF8, &[("path", &shown)]));
    };
    match output.try_exists() {
        Ok(false) => {}
        // A file standing where a parent directory should be: creating the
        // output below it is refused with what stands there.
        Err(error) if error.kind() == ErrorKind::NotADirectory => {}
        Ok(true) => {
            let state = file::state(output);
            return Err(refuse(&causes::OUTPUT_EXISTS, &[("path", &shown), ("state", &state)]));
        }
        Err(error) => {
            return Err(refuse(
                &causes::OUTPUT_UNINSPECTABLE,
                &[
                    ("path", &shown),
                    ("parent", &parent_shown),
                    ("parent_state", &file::state(&parent)),
                    ("kind", &file::kind(&error)),
                    ("error", &error.to_string()),
                ],
            ));
        }
    }
    let input = spec::Resolved::read(declaration)?;
    let tensors = weights::build(&input)?;
    let tensor_count = tensors.len();
    let configuration = crate::json::value(&input.dimensions);
    let declaration = crate::json::value(&input.spec);
    let uncreated = |what: &str, path: &Path, error: std::io::Error| {
        let above = path.parent().map_or_else(PathBuf::new, Path::to_path_buf);
        let cause = if error.kind() == ErrorKind::PermissionDenied {
            &causes::OUTPUT_DENIED
        } else {
            &causes::OUTPUT_UNCREATABLE
        };
        refuse(
            cause,
            &[
                ("what", what),
                ("path", &path.display().to_string()),
                ("parent", &above.display().to_string()),
                ("parent_state", &file::state(&above)),
                ("kind", &file::kind(&error)),
                ("error", &error.to_string()),
            ],
        )
    };
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(&parent).map_err(|error| uncreated("output parent", &parent, error))?;
    }
    fs::create_dir(output).map_err(|error| uncreated("output directory", output, error))?;
    let mut owned_files = Vec::new();
    let result = (|| -> Result<FixtureReport> {
        let write = |what: &str, path: &Path, bytes: &[u8]| {
            fs::write(path, bytes).map_err(|error| unwritten(what, path, bytes.len(), &error))
        };
        // Native checkpoint filenames, retained as external wire identities:
        // https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained
        let config = output.join("config.json");
        owned_files.push(config.clone());
        write("config", &config, &input.config_bytes)?;
        let tokenizer = output.join("tokenizer.json");
        owned_files.push(tokenizer.clone());
        write("tokenizer", &tokenizer, &input.tokenizer_bytes)?;
        let checkpoint = output.join("model.safetensors");
        owned_files.push(checkpoint.clone());
        let bytes: usize = tensors.values().map(|tensor| tensor.elem_count() * 4).sum();
        candle_core::safetensors::save(&tensors, &checkpoint).map_err(|error| {
            // The system's own answer when the write itself failed, so its
            // kind (a full disk, a refused directory) reaches the refusal.
            match crate::failures::file::tensors::io(&error) {
                Some(io) => unwritten("weights", &checkpoint, bytes, io),
                None => unwritten("weights", &checkpoint, bytes, &std::io::Error::other(error.to_string())),
            }
        })?;
        // The real consumer maps the saved checkpoint; release the producer's
        // tensor buffers before loading it instead of retaining both copies.
        drop(tensors);
        // Loading and forwarding declare their own refusals; they pass through.
        let runtime = Runtime::load(output_text, None, DeviceChoice::Cpu)?;
        let verification = verify::activations(&runtime, &input.spec.verification, &shown)?;
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
                    Err(cause) => failures.push(format!(
                        "removing the fixture file {} also failed ({}: {cause}; it is {})",
                        path.display(),
                        crate::failures::file::kind(&cause),
                        crate::failures::file::state(&path)
                    )),
                }
            }
            if let Err(cause) = fs::remove_dir(output) {
                failures.push(format!(
                    "removing the output directory {} also failed ({}: {cause}; it is {})",
                    output.display(),
                    crate::failures::file::kind(&cause),
                    crate::failures::file::state(output)
                ));
            }
            if failures.is_empty() {
                Err(crate::failures::amend(
                    error,
                    format!("the output {} this run created was removed", output.display()),
                ))
            } else {
                Err(crate::failures::amend(
                    error,
                    format!(
                        "{}; `rm -r {}` removes what is left",
                        failures.join("; "),
                        output.display()
                    ),
                ))
            }
        }
    }
}

/// The refusal of writing `bytes` bytes of the fixture's `what` to `path`.
fn unwritten(what: &str, path: &Path, bytes: usize, error: &std::io::Error) -> anyhow::Error {
    refuse(
        &causes::FILE_UNWRITTEN,
        &[
            ("what", what),
            ("path", &path.display().to_string()),
            ("bytes", &bytes.to_string()),
            ("kind", &file::kind(error)),
            ("error", &error.to_string()),
        ],
    )
}
