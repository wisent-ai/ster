//! How choosing the device and the precision a model runs at fails
//! (src/runtime/device.rs), before any checkpoint is read.

use wisent_errors::{Cause, Code, Operation};

use super::DOCS;

pub static DEVICE_UNKNOWN: Cause = Cause {
    id: "device-unknown",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the device",
    cause: "unknown device {value}; expected cpu, metal, or cuda",
    observed: "--device {value}",
    remedy: "give --device cpu, metal or cuda",
    test: "tests/runtime/device.sh::an_unknown_device_is_refused",
};

pub static DEVICE_UNBUILT: Cause = Cause {
    id: "device-unbuilt",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "opening the {device} device",
    cause: "this Ster binary was built without the {device} feature",
    observed: "--device {device}",
    remedy: "run with --device cpu, or install the Ster build for this machine that carries {device}",
    test: "tests/runtime/device.sh::a_device_this_build_lacks_is_refused",
};

pub static DEVICE_UNOPENED: Cause = Cause {
    id: "device-unopened",
    code: Code::Config,
    usage: false,
    exit: None,
    failed: "opening the {device} device",
    cause: "this build carries the {device} backend, and the machine's {device} driver opened no device: no such GPU is present, or its driver refused (too old, or held by another process)",
    observed: "--device {device}; the driver answered: {error}",
    remedy: "run with --device cpu, or check that this machine's {device} driver answers",
    test: "tests/runtime/device.sh::a_device_that_does_not_open_is_refused",
};

pub static PRECISION_UNKNOWN: Cause = Cause {
    id: "precision-unknown",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "reading the precision",
    cause: "unknown precision {value}; expected f32, f16, or bf16",
    observed: "--precision {value}",
    remedy: "give --precision f32, f16 or bf16",
    test: "tests/runtime/device.sh::an_unknown_precision_is_refused",
};

pub static BF16_ON_CPU: Cause = Cause {
    id: "bf16-on-cpu",
    code: Code::Config,
    usage: true,
    exit: None,
    failed: "mapping the weights of {model} at bf16 on the cpu",
    cause: "bf16 has no CPU matmul kernel in this Candle build, so the run would load the whole checkpoint and then fail at its first projection",
    observed: "--precision bf16 --device cpu (cpu is the device when --device is left out); nothing of {model} was read",
    remedy: "use --precision f16 for half precision on cpu, or --device metal for bf16",
    test: "tests/runtime/device.sh::bf16_on_the_cpu_is_refused",
};

/// The device and the dtype a model's weights are mapped at.
pub static DEVICE: Operation = Operation {
    id: "ster.runtime.device",
    entries: &[],
    uses: &[],
    summary: "read --device and --precision and open the device the weights are mapped on, at the dtype it can multiply in",
    docs: DOCS,
    warnings: &[],
    causes: &[
        &DEVICE_UNKNOWN,
        &DEVICE_UNBUILT,
        &DEVICE_UNOPENED,
        &PRECISION_UNKNOWN,
        &BF16_ON_CPU,
    ],
};
