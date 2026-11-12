//! How reading or writing one of Ster's safetensors files fails once Candle
//! holds it: a file that is not safetensors, a tensor stored in a dtype
//! Candle does not load, and tensors the device does not take or give back.
//! The file's own read or write failing is [`super::read`]'s or
//! [`super::write`]'s cause.

use std::path::Path;

use candle_core::Error;
use wisent_errors::{Cause, Code};

use super::super::refuse;
use super::state;

/// The I/O error under Candle's wrappers, when one is what failed.
pub fn io(error: &Error) -> Option<&std::io::Error> {
    match error {
        Error::Io(error) => Some(error),
        Error::WithBacktrace { inner, .. }
        | Error::Context { inner, .. }
        | Error::WithPath { inner, .. } => io(inner),
        _ => None,
    }
}

/// The safetensors error under Candle's wrappers, by what it says.
fn kind(error: &Error) -> &'static Cause {
    match error {
        Error::SafeTensor(_) => &MALFORMED,
        Error::UnsupportedSafeTensorDtype(_) => &DTYPE,
        Error::WithBacktrace { inner, .. }
        | Error::Context { inner, .. }
        | Error::WithPath { inner, .. } => kind(inner),
        _ => &UNPLACED,
    }
}

/// The refusal of reading `path`, the safetensors file holding `role`, when
/// Candle answered `error`.
pub fn read_refusal(role: &str, path: &Path, error: &Error) -> anyhow::Error {
    if let Some(failed) = io(error) {
        return super::read::refusal(role, path, failed);
    }
    refuse(
        kind(error),
        &[
            ("role", role),
            ("path", &path.display().to_string()),
            ("state", &state(path)),
            ("error", &error.to_string()),
        ],
    )
}

/// The refusal of writing `path`, the safetensors file holding `role`, when
/// Candle answered `error`; `left` says what the refused write left.
pub fn write_refusal(role: &str, path: &Path, left: &str, error: &Error) -> anyhow::Error {
    if let Some(failed) = io(error) {
        return super::write::refusal("writing", role, path, left, failed);
    }
    refuse(
        &UNSERIALIZED,
        &[
            ("role", role),
            ("path", &path.display().to_string()),
            ("left", left),
            ("error", &error.to_string()),
        ],
    )
}

pub static MALFORMED: Cause = Cause {
    id: "tensors-not-safetensors",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "{path} is not a safetensors file: its header or its table of tensors does not parse, so it was cut short while written or is a file of another format",
    observed: "path: {path} ({state}); the safetensors reader answered: {error}",
    remedy: "name {role} as Ster wrote it; a file cut short is written again by the command that made it",
    test: "tests/tuning/runs/adapters.sh::adapter_weights_that_are_not_safetensors_are_refused",
};

pub static DTYPE: Cause = Cause {
    id: "tensors-dtype-unloadable",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "a tensor of {path} is stored in a dtype Candle does not load, such as bool, i8 or u64 (Candle loads the integer, float and 8-, 6- and 4-bit float dtypes)",
    observed: "path: {path} ({state}); Candle answered: {error}",
    remedy: "name {role} as Ster wrote it; a file another tool wrote must hold its tensors in a dtype Candle loads",
    test: "tests/tuning/runs/adapters.sh::adapter_weights_of_a_dtype_candle_does_not_load_are_refused",
};

pub static UNPLACED: Cause = Cause {
    id: "tensors-unplaced",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "reading {role} {path}",
    cause: "the device did not take the tensors {path} holds: it has no memory left for them, or does not hold their dtype",
    observed: "path: {path} ({state}); the device answered: {error}",
    remedy: "run on a device with room for {role} (`--device cpu` uses host memory), or free the memory another process holds",
    test: "tests/tuning/runs/adapters.sh::adapter_weights_the_device_does_not_take_are_refused",
};

pub static UNSERIALIZED: Cause = Cause {
    id: "tensors-unserialized",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "writing {role} {path}",
    cause: "the tensors could not be laid out as safetensors: a tensor's data could not be copied off the device it lives on",
    observed: "path: {path}; {left}; Candle answered: {error}",
    remedy: "run the command again on `--device cpu`, where every tensor is in host memory",
    test: "tests/tuning/runs/adapters.sh::tensors_the_device_does_not_give_back_are_refused",
};
