//! Every value a request may leave out, and the reason each one is what it
//! is. Only choices with a neutral or documented meaning live here; every
//! training, sampling and filtering number is the caller's to state, and a
//! request without one is refused by name.

use ster::{ChatChoice, DeviceChoice, Precision, TrainingMethod};

/// The CPU, the one device every build has; `--device` defaults to it too.
pub(super) fn default_device() -> DeviceChoice {
    DeviceChoice::Cpu
}

pub(super) fn default_layers() -> String {
    "all".to_owned()
}

/// The mean difference, the method `ster vector train` defaults to.
pub(super) fn default_method() -> TrainingMethod {
    TrainingMethod::Caa
}

/// Apply the model's own conversation format when it publishes one. An
/// instruct checkpoint is the common case and raw text is wrong for it, so
/// the default is the setting that is right more often; `off` restores the
/// raw-text encoding a base model wants.
pub(super) fn default_chat_template() -> ChatChoice {
    ChatChoice::Auto
}

/// Single precision, which is what every recorded run used. Half precision is
/// opt-in because it changes the numbers a client may be comparing against.
pub(super) fn default_precision() -> Precision {
    Precision::F32
}
