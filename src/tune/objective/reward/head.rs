//! The scalar head itself: how it is created, how it scores a sequence, and
//! the loaded judge a policy run reaches for.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Init, VarMap};

use crate::{
    lora,
    runtime::{DeviceChoice, Runtime},
};

/// The scalar head a reward model scores with: one row, `hidden_size` wide.
#[derive(Debug, Clone)]
pub struct RewardHead {
    weight: Tensor,
}

impl RewardHead {
    /// A fresh head registered in `varmap`, so one optimizer steps it and the
    /// adapters together.
    ///
    /// Zeroed rather than drawn. A single output row has no symmetry for a
    /// random draw to break, and the Bradley-Terry gradient at zero is
    /// `-(h_chosen - h_rejected) / 2`, which is as far from zero as the two
    /// residual states are from each other — so the head learns immediately
    /// and the run needs no seed of its own.
    pub fn fresh(varmap: &VarMap, hidden: usize, device: &Device, dtype: DType) -> Result<Self> {
        let weight = varmap
            .get(
                (1, hidden),
                lora::REWARD_HEAD_TENSOR,
                Init::Const(0.0),
                dtype,
                device,
            )
            .with_context(|| format!("failed to create {}", lora::REWARD_HEAD_TENSOR))?;
        Ok(Self { weight })
    }

    /// The head read back out of an artifact, frozen.
    pub fn from_tensor(weight: Tensor) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("reward head has shape {dims:?}, expected one row of hidden-size weights");
        }
        Ok(Self { weight })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// The score of one sequence, given `[1, sequence, hidden]`.
    ///
    /// The last position is the one that has attended to the whole sequence,
    /// so it is the only position that can score it. A batched pass hands its
    /// rows over one at a time through `batch::row`, already sliced back to
    /// each row's own length, so the last position here is always a real token
    /// and never padding.
    pub fn score(&self, hidden: &Tensor) -> Result<Tensor> {
        let (_, sequence, width) = hidden.dims3()?;
        let expected = self.weight.dim(1)?;
        if width != expected {
            bail!("reward head is {expected} wide, the model's residual stream is {width}");
        }
        // Read at the head's own dtype rather than the residual stream's. The
        // head is the smallest parameter in the run — one row — and it is the
        // one that rounds away first, so it is trained in F32 even when the
        // base weights are half. At F32 this is a clone.
        let last = hidden
            .i((0, sequence - 1, ..))?
            .to_dtype(self.weight.dtype())?;
        // An elementwise product folded to a scalar rather than a matmul: the
        // result is one number, and a `[1, hidden] x [hidden, 1]` matmul would
        // reshape twice to say the same thing.
        Ok((last * self.weight.squeeze(0)?)?.sum_all()?)
    }
}

/// A trained reward model, loaded and frozen: the base weights with the
/// artifact's adapters attached, and the head that reads them.
///
/// This is a second model in memory beside whatever policy is being trained,
/// and that is not an oversight to optimize away later: a judge is genuinely a
/// different model from the thing it judges. What it is not is a second copy
/// of anything — the adapters are the artifact's own, and the base weights are
/// mapped read-only exactly as every other Ster load maps them.
pub struct RewardModel {
    runtime: Runtime,
    head: RewardHead,
}

impl RewardModel {
    /// Loads the reward artifact at `path` against `model`.
    ///
    /// The artifact must declare kind `reward`; a generation adapter has no
    /// head, and attaching one here would silently score every sequence with
    /// whatever the caller passed instead.
    pub fn load(
        model: &str,
        revision: Option<&str>,
        device: DeviceChoice,
        path: &Path,
    ) -> Result<Self> {
        let (runtime, artifact) =
            Runtime::load_artifact(model, revision, device, path, lora::Kind::Reward)?;
        let weight = artifact
            .tensors
            .get(lora::REWARD_HEAD_TENSOR)
            .with_context(|| format!("reward artifact is missing {}", lora::REWARD_HEAD_TENSOR))?
            .to_device(runtime.device())?
            .to_dtype(runtime.dtype())?;
        Ok(Self {
            runtime,
            head: RewardHead::from_tensor(weight)?,
        })
    }

    /// The reward this model assigns to one tokenized sequence.
    ///
    /// The whole sequence goes in, prompt included: the head reads the last
    /// position, which has attended to everything before it, so a response is
    /// scored in the context it was a response to.
    pub fn score(&self, ids: &[u32]) -> Result<f64> {
        let hidden = self.runtime.forward_hidden_scored(ids)?;
        Ok(self.head.score(&hidden)?.to_scalar::<f32>()? as f64)
    }
}
