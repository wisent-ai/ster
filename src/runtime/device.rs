//! Where a run computes and in what precision: the two operator words, and
//! what each resolves to on this machine.

use anyhow::{bail, Result};
use candle_core::{DType, Device};

#[derive(Debug, Clone, Copy)]
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
            _ => bail!("unknown device {value:?}; expected cpu, metal, or cuda"),
        }
    }

    pub(super) fn resolve(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Metal => {
                #[cfg(feature = "metal")]
                { anyhow::Context::context(Device::new_metal(0), "failed to initialize Metal device") }
                #[cfg(not(feature = "metal"))]
                { bail!("this Ster binary was built without the metal feature") }
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                { anyhow::Context::context(Device::new_cuda(0), "failed to initialize CUDA device") }
                #[cfg(not(feature = "cuda"))]
                { bail!("this Ster binary was built without the cuda feature") }
            }
        }
    }
}

/// The dtype the frozen base weights are mapped at.
///
/// Not the dtype of anything a run trains: adapters, a reward head, and every
/// optimizer moment stay in F32 whatever this says. Half precision here buys
/// two bytes per base weight instead of four — a checkpoint that ships in
/// bf16 currently costs twice its own file size to load — and on Metal it
/// reaches the half-precision matmul path the hardware has and F32 does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            _ => bail!("unknown precision {value:?}; expected f32, f16, or bf16"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// The Candle dtype, or a refusal naming the combination that has no
    /// kernel.
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
    pub(super) fn dtype(self, device: &Device) -> Result<DType> {
        match (self, device) {
            (Self::F32, _) => Ok(DType::F32),
            (Self::F16, _) => Ok(DType::F16),
            (Self::Bf16, Device::Cpu) => bail!(
                "bf16 has no CPU matmul kernel in this Candle build; use --precision f16 for half precision on cpu, or --device metal for bf16"
            ),
            (Self::Bf16, _) => Ok(DType::BF16),
        }
    }
}
