//! How loading a model fails: finding the checkpoint, reading its config
//! into a decoder, mapping its weights, and attaching a trained artifact.

use std::path::Path;

use serde_json::Value;
use wisent_errors::{Cause, Operation};

use super::{refuse, refuse_from};

pub mod checkpoint;
#[path = "formats/hub.rs"]
pub mod hub;
pub mod decoder;
pub mod config;
#[path = "formats/t5.rs"]
pub mod t5;
pub mod tokenizer;
pub mod weights;

pub use hub::{download_refusal, repository_refusal};

/// docs/guide/errors/model.md, where these functionalities are listed.
const DOCS: &str = page!("model");

/// Why a family's reading cannot run a setting its config states, each kind
/// its own declared cause in [`config`].
#[derive(Debug, Clone, Copy)]
pub enum Unrunnable {
    /// A variant of the setting the family's decoder does not implement.
    Unimplemented,
    /// Two settings that disagree with each other.
    Contradictory,
    /// A value outside the range the decoder runs.
    OutOfRange,
    /// A part of the decoder turned on without a setting it is sized by.
    Incomplete,
    /// A setting not in the form its family spells it.
    Malformed,
}

impl Unrunnable {
    /// The declared cause of this kind of refusal.
    pub fn cause(self) -> &'static Cause {
        match self {
            Self::Unimplemented => &config::SETTINGS_UNIMPLEMENTED,
            Self::Contradictory => &config::SETTINGS_CONTRADICTORY,
            Self::OutOfRange => &config::SETTINGS_OUT_OF_RANGE,
            Self::Incomplete => &config::SETTINGS_INCOMPLETE,
            Self::Malformed => &config::SETTINGS_MALFORMED,
        }
    }
}

/// The refusal of a setting a family's reading cannot run, for the reason
/// `kind` names: `detail` is the sentence that names the config, the key and
/// what the decoder implements, `observed` what the config states for the
/// keys it names ([`stated`]).
pub fn settings(kind: Unrunnable, detail: String, observed: String) -> anyhow::Error {
    refuse(
        kind.cause(),
        &[("detail", &detail), ("observed", &observed)],
    )
}

/// What `config` states for each of `keys`, the observed state of a settings
/// refusal: `key: value` in the config's own JSON spelling, `key: absent` for
/// a key it does not state. A dotted key (`attn_config.alibi`) is read
/// through the objects it names.
pub fn stated(config: &Value, keys: &[&str]) -> String {
    keys.iter()
        .map(|key| {
            match key
                .split('.')
                .try_fold(config, |object, part| object.get(part))
            {
                Some(value) => format!("{key}: {value}"),
                None => format!("{key}: absent"),
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// What a family's reading of an optional key turns into its refusal:
/// `.or_refuse(raw, &[key], || sentence)` when the key is absent (or a value
/// the reading treats as absent, such as zero), `.or_malformed(…)` when it is
/// stated in another form than the family spells it; `raw` is the config
/// object the keys are read from.
pub trait Stated<T> {
    fn or_refuse(
        self,
        config: &Value,
        keys: &[&str],
        detail: impl FnOnce() -> String,
    ) -> anyhow::Result<T>;

    fn or_malformed(
        self,
        config: &Value,
        keys: &[&str],
        detail: impl FnOnce() -> String,
    ) -> anyhow::Result<T>;
}

impl<T> Stated<T> for Option<T> {
    fn or_refuse(
        self,
        config: &Value,
        keys: &[&str],
        detail: impl FnOnce() -> String,
    ) -> anyhow::Result<T> {
        self.ok_or_else(|| settings(Unrunnable::Incomplete, detail(), stated(config, keys)))
    }

    fn or_malformed(
        self,
        config: &Value,
        keys: &[&str],
        detail: impl FnOnce() -> String,
    ) -> anyhow::Result<T> {
        self.ok_or_else(|| settings(Unrunnable::Malformed, detail(), stated(config, keys)))
    }
}

/// What a model config is called in a refusal of its file.
pub const CONFIG_ROLE: &str = "the model config";

/// The refusal of reading the config file at `path`.
pub fn config_unreadable(path: &Path, error: &std::io::Error) -> anyhow::Error {
    super::file::read::refusal(CONFIG_ROLE, path, error)
}

/// The refusal of the config file at `path` the JSON reader stopped in.
pub fn config_malformed(path: &Path, error: &serde_json::Error) -> anyhow::Error {
    super::file::json::refusal(CONFIG_ROLE, config::FORM, path, error)
}

/// The refusal of the config file at `path` that is JSON and holds `held`
/// where one object belongs.
pub fn config_not_object(path: &Path, held: &Value) -> anyhow::Error {
    refuse(
        &config::NOT_OBJECT,
        &[
            ("path", &path.display().to_string()),
            ("held", &wisent_errors::trim_detail(&held.to_string(), 80)),
        ],
    )
}

/// The refusal of weights the decoder `model`'s config describes could not
/// be built from: a tensor of another shape, a tensor the checkpoint lacks,
/// or the device refusing them; its observed state the tensor and both
/// shapes Candle compared. A cause the build declared itself
/// (`refuse_candle`) is that refusal.
pub fn weights_unfit(model: &str, error: candle_core::Error) -> anyhow::Error {
    let error = match super::lift(error) {
        Ok(declared) => return declared,
        Err(error) => error,
    };
    let cause = match super::candle_root(&error) {
        candle_core::Error::UnexpectedShape { .. } => &weights::WEIGHTS_UNFIT,
        candle_core::Error::CannotFindTensor { .. } => &weights::TENSOR_UNSTORED,
        _ => &weights::WEIGHTS_UNBUILT,
    };
    refuse_from(
        cause,
        &[
            ("model", model),
            ("error", &error.to_string()),
            ("shapes", &unfit(&error)),
        ],
        error.into(),
    )
}

/// What Candle found wrong building a decoder: its own message naming the
/// tensor, with the shape the config gives it and the shape the checkpoint
/// stores, the tensor the checkpoint lacks, or Candle's own sentence for
/// anything else.
fn unfit(error: &candle_core::Error) -> String {
    match error {
        candle_core::Error::UnexpectedShape { msg, expected, got } => format!(
            "Candle: {msg}; the config sizes it {:?}, the checkpoint stores {:?}",
            expected.dims(),
            got.dims()
        ),
        candle_core::Error::CannotFindTensor { path } => {
            format!("tensor {path}: the checkpoint holds no tensor of that name")
        }
        candle_core::Error::WithBacktrace { inner, .. }
        | candle_core::Error::Context { inner, .. }
        | candle_core::Error::WithPath { inner, .. } => unfit(inner),
        other => format!("Candle's {} error: {other}", super::tensor::variant(other)),
    }
}

pub(super) static OPERATIONS: &[&Operation] = &[
    &checkpoint::CHECKPOINT,
    &config::CONFIG,
    &tokenizer::TOKENIZER,
    &decoder::DECODER,
    &t5::T5,
    &weights::WEIGHTS,
    &weights::LOAD,
    &weights::adapters::ADAPTERS,
];
