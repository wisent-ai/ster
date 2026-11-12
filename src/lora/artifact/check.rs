//! Every check a reader runs on an adapter artifact before trusting it: what
//! its sidecar declares, and that every tensor it declares is there, shaped
//! and finite.

use anyhow::Result;
use candle_core::{DType, Tensor};

use super::super::{Widths, adapter::Adapter};
use super::{ARTIFACT_SCHEMA_VERSION, Artifact, Kind, REWARD_HEAD_TENSOR};
use crate::failures::{
    lora::artifact::{
        ALPHA_INVALID, HEAD_MISSING, HEAD_UNEXPECTED, HIDDEN_SIZE_ZERO, LAYER_REPEATED,
        LAYERS_NONE, PRODUCT_FOREIGN, RANK_ZERO, SCHEMA_UNSUPPORTED, TARGET_REPEATED, TARGETS_NONE,
        TENSOR_ABSENT, TENSOR_NOT_FINITE, TENSOR_SHAPE, WIDTH_UNFIT,
    },
    refuse,
    tune::run::score_failed,
};

impl Artifact {
    /// Refuses malformed artifacts using the dimensions recorded in the sidecar.
    ///
    /// Attention and feed-forward widths come from the checkpoint, not hidden_size.
    /// Query outputs need not have the residual width.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ARTIFACT_SCHEMA_VERSION {
            return Err(refuse(
                &SCHEMA_UNSUPPORTED,
                &[
                    ("schema", &self.schema_version.to_string()),
                    ("supported", &ARTIFACT_SCHEMA_VERSION.to_string()),
                ],
            ));
        }
        if self.product != "ster" {
            return Err(refuse(
                &PRODUCT_FOREIGN,
                &[("product", &format!("{:?}", self.product))],
            ));
        }
        let targets: Vec<&str> = self.targets.iter().map(|target| target.name()).collect();
        let listed: Vec<String> = self.layers.iter().map(usize::to_string).collect();
        if self.rank == 0 {
            return Err(refuse(&RANK_ZERO, &[]));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(refuse(
                &ALPHA_INVALID,
                &[("alpha", &self.alpha.to_string()), ("rank", &self.rank.to_string())],
            ));
        }
        if self.hidden_size == 0 {
            return Err(refuse(&HIDDEN_SIZE_ZERO, &[("model", &self.model)]));
        }
        if self.targets.is_empty() {
            return Err(refuse(&TARGETS_NONE, &[("layers", &listed.join(","))]));
        }
        if self.layers.is_empty() {
            return Err(refuse(&LAYERS_NONE, &[("targets", &targets.join(","))]));
        }
        if let Some(target) = targets
            .iter()
            .enumerate()
            .find_map(|(index, target)| targets[..index].contains(target).then_some(target))
        {
            return Err(refuse(
                &TARGET_REPEATED,
                &[("target", target), ("targets", &targets.join(","))],
            ));
        }
        if let Some(layer) = self
            .layers
            .iter()
            .enumerate()
            .find_map(|(index, layer)| self.layers[..index].contains(layer).then_some(layer))
        {
            return Err(refuse(
                &LAYER_REPEATED,
                &[("layer", &layer.to_string()), ("layers", &listed.join(","))],
            ));
        }
        for layer in &self.layers {
            for target in &self.targets {
                let (a_name, b_name) = Adapter::tensor_names(*layer, *target);
                let a = self.tensor(&a_name)?;
                let b = self.tensor(&b_name)?;
                check_factor(&a_name, a, 0, self.rank)?;
                if target.reads_hidden() {
                    check_factor(&a_name, a, 1, self.hidden_size)?;
                }
                check_factor(&b_name, b, 1, self.rank)?;
                if target.writes_hidden() {
                    check_factor(&b_name, b, 0, self.hidden_size)?;
                }
                for (name, tensor) in [(&a_name, a), (&b_name, b)] {
                    finite(name, tensor)?;
                }
            }
        }
        // The head is the whole point of a reward artifact and meaningless in
        // a plain one, so both directions are refused: a reward document
        // without a head would load as a generation adapter that scores
        // nothing, and an adapter carrying one would have come from somewhere
        // this build cannot account for.
        match (self.kind, self.tensors.get(REWARD_HEAD_TENSOR)) {
            (Kind::Reward, Some(head)) => {
                check_factor(REWARD_HEAD_TENSOR, head, 0, 1)?;
                check_factor(REWARD_HEAD_TENSOR, head, 1, self.hidden_size)?;
                finite(REWARD_HEAD_TENSOR, head)?;
            }
            (Kind::Reward, None) => {
                let count = self.tensors.len().to_string();
                return Err(refuse(&HEAD_MISSING, &[("count", &count)]));
            }
            (Kind::Adapter, Some(head)) => {
                return Err(refuse(
                    &HEAD_UNEXPECTED,
                    &[
                        ("shape", &format!("{:?}", head.dims())),
                        ("count", &self.tensors.len().to_string()),
                        ("model", &self.model),
                    ],
                ));
            }
            (Kind::Adapter, None) => {}
        }
        Ok(())
    }

    /// The tensor `name` the sidecar's targets and layers need.
    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| refuse(&TENSOR_ABSENT, &[("name", name)]))
    }

    /// Checks the remaining factor axes against the checkpoint before attaching it.
    pub(crate) fn validate_widths(&self, widths: Widths) -> Result<()> {
        for &layer in &self.layers {
            for &target in &self.targets {
                let (outputs, inputs) = target.widths(widths);
                let (a, b) = Adapter::tensor_names(layer, target);
                for (name, expected) in [(a, [self.rank, inputs]), (b, [outputs, self.rank])] {
                    let tensor = self.tensor(&name)?;
                    if tensor.dims() != expected {
                        return Err(refuse(
                            &WIDTH_UNFIT,
                            &[
                                ("name", &name),
                                ("found", &format!("{:?}", tensor.dims())),
                                ("expected", &format!("{expected:?}")),
                            ],
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Checks one dimension of one factor, naming the axis the way a reader thinks of it.
fn check_factor(name: &str, tensor: &Tensor, axis: usize, expected: usize) -> Result<()> {
    let dims = tensor.dims();
    let shape = format!("{dims:?}");
    if dims.len() != 2 {
        return Err(refuse(
            &TENSOR_SHAPE,
            &[
                ("name", name),
                ("found", &format!("{} dimensions", dims.len())),
                ("expected", "2"),
                ("shape", &shape),
            ],
        ));
    }
    if dims[axis] != expected {
        let axis_name = if axis == 0 { "rows" } else { "columns" };
        return Err(refuse(
            &TENSOR_SHAPE,
            &[
                ("name", name),
                ("found", &format!("{} {axis_name}", dims[axis])),
                ("expected", &expected.to_string()),
                ("shape", &shape),
            ],
        ));
    }
    Ok(())
}

/// Refuses a tensor with a value no forward pass could survive.
///
/// A non-finite factor does not fail loudly at matmul time: it propagates a
/// `NaN` through the residual stream and comes out as a plausible-looking
/// decode, so it is caught when the document is read rather than when the
/// damage is visible.
fn finite(name: &str, tensor: &Tensor) -> Result<()> {
    let values = tensor
        .flatten_all()
        .and_then(|flat| flat.to_dtype(DType::F32))
        .and_then(|flat| flat.to_vec1::<f32>())
        .map_err(|error| score_failed(&format!("adapter tensor {name}"), error.into()))?;
    let count = values.iter().filter(|value| !value.is_finite()).count();
    if count > 0 {
        return Err(refuse(
            &TENSOR_NOT_FINITE,
            &[("name", name), ("count", &count.to_string())],
        ));
    }
    Ok(())
}
