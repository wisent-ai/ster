//! A steering plan and what one forward pass returns.

use std::collections::BTreeMap;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct SteeringPlan {
    vectors: BTreeMap<usize, Tensor>,
    pub(super) strength: f64,
    pub(super) hidden_size: usize,
}

/// Every plan is built from vectors `Runtime::check_steering` or BiPO's own
/// variable already fitted to the model: at least one, each as wide as the
/// residual stream. Another shape is a defect of the caller.
const FITTED: &str = "a steering plan holds at least one vector as wide as the residual stream";

impl SteeringPlan {
    pub fn new(
        vectors: impl IntoIterator<Item = (usize, Vec<f32>)>,
        strength: f64,
        hidden_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (layer, values) in vectors {
            assert_eq!(values.len(), hidden_size, "{FITTED}");
            let tensor = Tensor::from_vec(values, hidden_size, device)?.to_dtype(dtype)?;
            tensors.insert(layer, tensor);
        }
        assert!(!tensors.is_empty(), "{FITTED}");
        Ok(Self {
            vectors: tensors,
            strength,
            hidden_size,
        })
    }

    /// A plan over tensors the caller already holds, cast to the model's
    /// `dtype`. The cast and the decoder's `vector * strength` both record
    /// backward nodes, so a plan built over a trainable variable carries the
    /// loss's gradient back to it — which is what a steering vector learned
    /// by optimization (`ster tune bipo`) needs and `new`'s copy of plain
    /// values cannot give.
    pub fn from_tensors(
        vectors: impl IntoIterator<Item = (usize, Tensor)>,
        strength: f64,
        hidden_size: usize,
        dtype: DType,
    ) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (layer, tensor) in vectors {
            assert_eq!(tensor.dims(), [hidden_size], "{FITTED}");
            tensors.insert(layer, tensor.to_dtype(dtype)?);
        }
        assert!(!tensors.is_empty(), "{FITTED}");
        Ok(Self {
            vectors: tensors,
            strength,
            hidden_size,
        })
    }

    pub(super) fn vector(&self, layer: usize) -> Option<&Tensor> {
        self.vectors.get(&layer)
    }
}

#[derive(Debug)]
pub struct ForwardOutput {
    /// The vocabulary projection the readout asked for, or `None` when it
    /// asked for none.
    ///
    /// A reward model reads the residual stream and never touches the
    /// vocabulary; on a real checkpoint that projection is the widest matmul
    /// in the pass, so skipping it is worth an `Option` at the two call sites
    /// that unwrap one.
    pub logits: Option<Tensor>,
    /// The residual stream after the final norm, `[batch, sequence, hidden]`.
    ///
    /// Always returned, because a `Tensor` is a handle and returning it costs
    /// a refcount rather than a copy.
    pub hidden: Tensor,
    /// Each captured layer's last-position state, as `train`, `extract` and
    /// every pair read take it.
    pub activations: BTreeMap<usize, Vec<f32>>,
    /// Each captured layer's state at every position, `[sequence, hidden]`,
    /// for a read that scores a text token by token (`ster vector score`).
    /// Empty from a pass that keeps none.
    pub positions: BTreeMap<usize, Tensor>,
}

impl ForwardOutput {
    /// The vocabulary projection of a pass run with `readout`, refused as
    /// `ster.model.decoder.logits-unread` when that readout projected none.
    pub fn into_logits(self, readout: super::Readout) -> anyhow::Result<Tensor> {
        match self.logits {
            Some(logits) => Ok(logits),
            None => Err(crate::failures::refuse(
                &crate::failures::model::decoder::LOGITS_UNREAD,
                &[
                    ("readout", &format!("{readout:?}")),
                    ("hidden", &format!("{:?}", self.hidden.dims())),
                ],
            )),
        }
    }
}
