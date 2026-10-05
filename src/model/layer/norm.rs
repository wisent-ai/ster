//! The normalisations the supported families use, as one type.
//!
//! Llama, Qwen, Gemma and their kin divide by the root mean square; StableLM,
//! Starcoder2, Phi-2, Cohere and Nemotron subtract the mean first (LayerNorm),
//! some with a bias and some without. Gemma and Nemotron store the scale as an
//! offset from one. Cohere and StableLM give query and key a separate scale per
//! head, so a weight may be `[heads, head_dim]` rather than one vector.

use candle_core::{D, DType, Tensor};
use candle_nn::VarBuilder;

use crate::model::{Architecture, NormKind, Pass};

#[derive(Debug, Clone)]
pub(in crate::model) struct Norm {
    /// `[width]`, or `[heads, head_dim]` for a per-head query or key norm.
    weight: Tensor,
    bias: Option<Tensor>,
    kind: NormKind,
    eps: f64,
    /// How many equal groups of the last axis are normalised apart
    /// (K2-Horizon's `layernorm_num_groups`); one for a plain norm.
    groups: usize,
}

/// What every norm of one checkpoint shares: its kind, its epsilon, whether
/// the stored scale is an offset from one, and how many equal groups a
/// hidden-width norm splits its input into.
#[derive(Debug, Clone, Copy)]
pub(in crate::model) struct NormSpec {
    pub kind: NormKind,
    pub eps: f64,
    pub offset: bool,
    pub groups: usize,
}

impl NormSpec {
    pub fn of(architecture: &Architecture) -> Self {
        Self {
            kind: architecture.norm,
            eps: architecture.norm_eps,
            offset: architecture.norm_offset,
            groups: architecture.norm_groups,
        }
    }

    /// One norm over `width`, from `builder`'s `weight` (and `bias`). A bare
    /// norm stores nothing, so its scale is a constant one.
    pub fn load(self, width: usize, builder: VarBuilder<'_>) -> candle_core::Result<Norm> {
        let weight = match self.kind {
            NormKind::Bare => Tensor::ones(width, builder.dtype(), builder.device())?,
            _ => builder.get(width, "weight")?,
        };
        let bias = self.bias(|| builder.get(width, "bias"))?;
        self.assemble(weight, bias)
    }

    /// A norm over `width` that stores no scale, so its scale is a constant
    /// one (Gemma 4's `v_norm` and its router's norm).
    pub fn unscaled(self, width: usize, builder: &VarBuilder<'_>) -> candle_core::Result<Norm> {
        Ok(Norm {
            weight: Tensor::ones(width, builder.dtype(), builder.device())?,
            bias: None,
            kind: self.kind,
            eps: self.eps,
            groups: self.groups,
        })
    }

    /// A per-head norm whose scale is one `[heads, head_dim]` tensor
    /// (Cohere's `q_norm` and `k_norm`).
    pub fn load_head_weights(
        self,
        heads: usize,
        head_dim: usize,
        builder: VarBuilder<'_>,
    ) -> candle_core::Result<Norm> {
        let weight = builder.get((heads, head_dim), "weight")?;
        let bias = self.bias(|| builder.get((heads, head_dim), "bias"))?;
        self.assemble(weight, bias)
    }

    /// A per-head norm stored as one module per head (StableLM's
    /// `q_layernorm.norms.{h}`), stacked into a `[heads, head_dim]` scale.
    pub fn load_head_modules(
        self,
        heads: usize,
        head_dim: usize,
        builder: VarBuilder<'_>,
    ) -> candle_core::Result<Norm> {
        let norms = builder.pp("norms");
        let stacked = |name: &str| -> candle_core::Result<Tensor> {
            let parts = (0..heads)
                .map(|head| norms.pp(head.to_string()).get(head_dim, name))
                .collect::<candle_core::Result<Vec<_>>>()?;
            Tensor::stack(&parts, 0)
        };
        let weight = stacked("weight")?;
        let bias = self.bias(|| stacked("bias"))?;
        self.assemble(weight, bias)
    }

    fn bias(
        self,
        load: impl FnOnce() -> candle_core::Result<Tensor>,
    ) -> candle_core::Result<Option<Tensor>> {
        match self.kind {
            NormKind::Layer { bias: true } => load().map(Some),
            _ => Ok(None),
        }
    }

    /// The shifted scale is a new tensor, never a variable, so the base stays
    /// frozen exactly as before.
    fn assemble(self, weight: Tensor, bias: Option<Tensor>) -> candle_core::Result<Norm> {
        Ok(Norm {
            weight: if self.offset { (weight + 1.0)? } else { weight },
            bias,
            kind: self.kind,
            eps: self.eps,
            groups: self.groups,
        })
    }
}

impl Norm {
    /// This norm with its scale and bias multiplied by `factor`, so its
    /// whole output is (LongCat-Flash's latent-attention scaling).
    pub fn scaled(self, factor: f64) -> candle_core::Result<Self> {
        Ok(Self {
            weight: (self.weight * factor)?,
            bias: self.bias.map(|bias| bias * factor).transpose()?,
            ..self
        })
    }
}

impl Norm {
    /// Normalises the last axis: fused for inference where Candle has a
    /// kernel, composed for training.
    ///
    /// `candle_nn::ops::rms_norm` and `candle_nn::ops::layer_norm` both end in
    /// `apply_op*_no_bwd` (candle-nn-0.11.0/src/ops.rs), so neither records a
    /// node the autograd tape can walk back through. The composed form is the
    /// one `candle_nn::LayerNorm::forward` falls back to
    /// (candle-nn-0.11.0/src/layer_norm.rs:123-142): widen half precision to
    /// F32, subtract the mean when the kind asks, divide by the root of the
    /// mean square plus `eps`, cast back, scale, and add the bias. A per-head
    /// weight broadcasts over the leading axes the same way. A grouped norm
    /// normalises each of its equal groups of the last axis apart, then
    /// scales the whole axis.
    pub fn forward(&self, hidden: &Tensor, pass: Pass) -> candle_core::Result<Tensor> {
        let fused = pass == Pass::Inference && self.weight.rank() == 1 && hidden.is_contiguous() && self.groups == 1;
        if fused {
            match (self.kind, &self.bias) {
                (NormKind::Rms, None) => {
                    return candle_nn::ops::rms_norm(hidden, &self.weight, self.eps as f32);
                }
                (NormKind::Layer { .. }, Some(bias)) => {
                    return candle_nn::ops::layer_norm(hidden, &self.weight, bias, self.eps as f32);
                }
                _ => {}
            }
        }
        let dtype = hidden.dtype();
        let internal = match dtype {
            DType::F16 | DType::BF16 => DType::F32,
            other => other,
        };
        let shape = hidden.shape().clone();
        let hidden = if self.groups > 1 {
            let mut grouped = shape.dims().to_vec();
            let width = grouped.pop().unwrap_or(0);
            grouped.extend([self.groups, width / self.groups]);
            hidden.reshape(grouped)?
        } else {
            hidden.clone()
        };
        let width = hidden.dim(D::Minus1)? as f64;
        let hidden = hidden.to_dtype(internal)?;
        let hidden = match self.kind {
            NormKind::Rms => hidden,
            NormKind::Layer { .. } | NormKind::Bare => {
                let mean = (hidden.sum_keepdim(D::Minus1)? / width)?;
                hidden.broadcast_sub(&mean)?
            }
        };
        let square = (hidden.sqr()?.sum_keepdim(D::Minus1)? / width)?;
        let normed = hidden.broadcast_div(&(square + self.eps)?.sqrt()?)?.reshape(shape)?;
        let scaled = normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            Some(bias) => scaled.broadcast_add(bias),
            None => Ok(scaled),
        }
    }
}
