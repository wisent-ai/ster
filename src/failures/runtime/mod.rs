//! How a model's runtime fails around its weights: the device and precision
//! they are mapped at (src/runtime/device.rs), turning text into the token
//! ids this checkpoint expects (src/runtime/text.rs), steering it with an
//! artifact and sampling a continuation (src/runtime/infer).

use wisent_errors::Operation;

/// docs/guide/errors/runtime.md, where these functionalities are listed.
const DOCS: &str = page!("runtime");

pub mod device;
pub mod encode;
pub mod generate;
pub mod steer;

pub(super) static OPERATIONS: &[&Operation] = &[
    &device::DEVICE,
    &encode::ENCODE,
    &steer::LAYERS,
    &steer::STEER,
    &generate::GENERATE,
    &generate::COMMAND,
];
