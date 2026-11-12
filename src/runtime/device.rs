//! Where a run computes and in what precision: the two operator words, and
//! what each resolves to on this machine.

use anyhow::Result;
use candle_core::{DType, Device};
use serde::Deserialize;

use crate::failures::{
    refuse,
    runtime::device::{BF16_ON_CPU, DEVICE_UNKNOWN, PRECISION_UNKNOWN},
};

/// The refusal of `device`, which this binary was built without.
#[cfg(any(not(feature = "metal"), not(feature = "cuda")))]
fn unbuilt(device: &str) -> anyhow::Error {
    refuse(
        &crate::failures::runtime::device::DEVICE_UNBUILT,
        &[("device", device)],
    )
}

/// The refusal of `device`, which this binary carries and could not open.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn unopened(device: &str, error: candle_core::Error) -> anyhow::Error {
    refuse(
        &crate::failures::runtime::device::DEVICE_UNOPENED,
        &[("device", device), ("error", &error.to_string())],
    )
}

/// A request body names it as the same word `--device` takes; an unknown one
/// is refused while the body is read, with the sentence `parse` writes.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(try_from = "String")]
pub enum DeviceChoice {
    Cpu,
    Metal,
    Cuda,
}

impl DeviceChoice {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(refuse(&DEVICE_UNKNOWN, &[("value", &format!("{value:?}"))])),
        }
    }

    pub(super) fn resolve(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Metal => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).map_err(|error| unopened("metal", error))
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(unbuilt("metal"))
                }
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).map_err(|error| unopened("cuda", error))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(unbuilt("cuda"))
                }
            }
        }
    }
}

impl TryFrom<String> for DeviceChoice {
    type Error = anyhow::Error;
    fn try_from(value: String) -> Result<Self> {
        Self::parse(&value)
    }
}

/// The dtype the frozen base weights are mapped at.
///
/// Not the dtype of anything a run trains: adapters, a reward head, and every
/// optimizer moment stay in F32 whatever this says. Half precision here buys
/// two bytes per base weight instead of four — a checkpoint that ships in
/// bf16 currently costs twice its own file size to load — and on Metal it
/// reaches the half-precision matmul path the hardware has and F32 does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum Precision {
    F32,
    F16,
    Bf16,
}

impl Precision {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            _ => Err(refuse(
                &PRECISION_UNKNOWN,
                &[("value", &format!("{value:?}"))],
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// The Candle dtype for `model`'s weights, or a refusal naming the
    /// combination that has no kernel.
    ///
    /// The one refusal here is bf16 on the CPU, and it is a fact about this
    /// Candle build rather than a policy: `cpu_backend`'s matmul accepts F16,
    /// F32 and F64 and returns `unsupported dtype BF16 for op matmul` for
    /// anything else, so a bf16 CPU run does not run slowly — it loads the
    /// whole checkpoint and then fails at the first projection. Saying so at
    /// the flag costs the operator seconds instead of minutes, and names the
    /// half precision that does work on the device they asked for. F16 on the
    /// CPU is real half arithmetic on this class of machine: `gemm` selects a
    /// native `neonfp16` microkernel on aarch64 when the hardware reports the
    /// `fp16` feature.
    pub(super) fn dtype(self, device: &Device, model: &str) -> Result<DType> {
        match (self, device) {
            (Self::F32, _) => Ok(DType::F32),
            (Self::F16, _) => Ok(DType::F16),
            (Self::Bf16, Device::Cpu) => Err(refuse(&BF16_ON_CPU, &[("model", model)])),
            (Self::Bf16, _) => Ok(DType::BF16),
        }
    }
}

impl TryFrom<String> for Precision {
    type Error = anyhow::Error;
    fn try_from(value: String) -> Result<Self> {
        Self::parse(&value)
    }
}
