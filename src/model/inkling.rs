//! Inkling's residual short convolution and feed-forward
//! (`modeling_inkling.py`, weights as Thinking Machines stores them below
//! `model.llm`).
//!
//! A residual short convolution adds to its input a depthwise causal
//! convolution of it over `kernel` positions, in F32, with no bias and no
//! activation. A decode step convolves over the last `kernel - 1` inputs of
//! the calls before it, kept in the cache.
//!
//! The dense feed-forward stores gate and up interleaved row by row in
//! `mlp.w13_dn` (gate on even rows) and multiplies its output by
//! `mlp.global_scale`. The routed one scores `n_routed_experts +
//! n_shared_experts` logits with `mlp.gate`; the top `k` routed experts by
//! `sigmoid(logit) + mlp.gate.bias` are chosen, and every chosen and shared
//! expert weighs `sigmoid(logit)` over the sum of theirs, times `route_scale`
//! and `mlp.gate.global_scale`. Experts store gate and up interleaved in
//! `experts.w13_weight` and `shared_experts.shared_w13_weight`.

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use super::{Activation, Cache, InklingSpec};

/// A residual short convolution and the slot its history is kept under.
#[derive(Debug, Clone)]
pub(super) struct ResidualConv {
    /// `[channels, kernel]` in F32.
    taps: Tensor,
    slot: usize,
}

impl ResidualConv {
    /// The taps stored at `name.weight`, `[channels, kernel]` or `[channels,
    /// 1, kernel]`.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        name: &str,
        channels: usize,
        kernel: usize,
        slot: usize,
    ) -> candle_core::Result<Self> {
        let stored = builder.pp(name).get_unchecked("weight")?;
        if stored.elem_count() != channels * kernel {
            candle_core::bail!(
                "{name}.weight holds {:?}, not {channels} channels of {kernel} taps",
                stored.dims()
            );
        }
        Ok(Self {
            taps: stored.reshape((channels, kernel))?.to_dtype(DType::F32)?,
            slot,
        })
    }

    /// `input + conv(input)`, `input` `[batch, sequence, channels]`.
    pub(super) fn forward(
        &self,
        input: &Tensor,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let dtype = input.dtype();
        let current = input.to_dtype(DType::F32)?;
        let (batch, sequence, channels) = current.dims3()?;
        let kernel = self.taps.dim(1)?;
        let key = (layer, self.slot);
        let history = match cache.convolutions.get(&key) {
            Some(history) if cache.use_kv_cache => history.clone(),
            _ => Tensor::zeros((batch, kernel - 1, channels), DType::F32, current.device())?,
        };
        let padded = Tensor::cat(&[&history, &current], 1)?;
        if cache.use_kv_cache {
            cache
                .convolutions
                .insert(key, padded.narrow(1, sequence, kernel - 1)?.contiguous()?);
        }
        let mut output = current.clone();
        for tap in 0..kernel {
            let weights = self.taps.narrow(1, tap, 1)?.squeeze(1)?;
            output = (output + padded.narrow(1, tap, sequence)?.broadcast_mul(&weights)?)?;
        }
        output.to_dtype(dtype)
    }
}

/// Gate and up from rows stored interleaved, gate on the even rows.
fn deinterleave(
    rows: &Tensor,
    intermediate: usize,
    hidden: usize,
) -> candle_core::Result<(Linear, Linear)> {
    let pairs = rows.reshape((intermediate, 2, hidden))?;
    Ok((
        Linear::new(pairs.i((.., 0, ..))?.contiguous()?, None),
        Linear::new(pairs.i((.., 1, ..))?.contiguous()?, None),
    ))
}

#[derive(Debug, Clone)]
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    fn forward(&self, input: &Tensor, activation: Activation) -> candle_core::Result<Tensor> {
        self.down
            .forward(&activation.gated(&self.gate.forward(input)?, &self.up.forward(input)?)?)
    }

    /// Expert `index` of the stacked `w13` `[experts, 2 · intermediate,
    /// hidden]` and `w2` `[experts, hidden, intermediate]`.
    fn stacked(
        w13: &Tensor,
        w2: &Tensor,
        index: usize,
        intermediate: usize,
        hidden: usize,
    ) -> candle_core::Result<Self> {
        let (gate, up) = deinterleave(&w13.get(index)?, intermediate, hidden)?;
        Ok(Self {
            gate,
            up,
            down: Linear::new(w2.get(index)?, None),
        })
    }
}

/// Inkling's feed-forward on one layer, before its residual convolution.
#[derive(Debug, Clone)]
enum Inner {
    Dense { expert: Expert, scale: Tensor },
    Routed(Box<Routed>),
}

#[derive(Debug, Clone)]
struct Routed {
    router: Linear,
    /// `mlp.gate.bias`, added to the routed scores to choose, on the host.
    bias: Vec<f32>,
    /// `route_scale · mlp.gate.global_scale`.
    scale: f32,
    experts: Vec<Expert>,
    shared: Vec<Expert>,
    top_k: usize,
}

/// One layer's feed-forward and the residual convolution over its output
/// (`mlp_sconv`, slot 3).
#[derive(Debug, Clone)]
pub(super) struct FeedForward {
    inner: Inner,
    conv: ResidualConv,
    activation: Activation,
}

impl FeedForward {
    /// `builder` is the layer's.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        intermediate: usize,
        spec: &InklingSpec,
        activation: Activation,
        layer: usize,
    ) -> candle_core::Result<Self> {
        let block = builder.pp("mlp");
        let inner = if spec.routed_layers >> layer & 1 == 1 {
            let (count, shared, width) = (spec.experts, spec.shared, spec.expert_intermediate);
            let gate = block.pp("gate");
            let w13 = block
                .pp("experts")
                .get((count, 2 * width, hidden), "w13_weight")?;
            let w2 = block
                .pp("experts")
                .get((count, hidden, width), "w2_weight")?;
            let shared_w13 = block
                .pp("shared_experts")
                .get((shared, 2 * width, hidden), "shared_w13_weight")?;
            let shared_w2 = block
                .pp("shared_experts")
                .get((shared, hidden, width), "shared_w2_weight")?;
            let global = gate
                .get_unchecked("global_scale")?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let [global] = global.as_slice() else {
                candle_core::bail!(
                    "layer {layer}'s mlp.gate.global_scale holds {} values, not one",
                    global.len()
                );
            };
            Inner::Routed(Box::new(Routed {
                router: linear_no_bias(hidden, count + shared, gate.clone())?,
                bias: gate.get(count, "bias")?.to_dtype(DType::F32)?.to_vec1()?,
                scale: spec.route_scale as f32 * global,
                experts: (0..count)
                    .map(|index| Expert::stacked(&w13, &w2, index, width, hidden))
                    .collect::<candle_core::Result<_>>()?,
                shared: (0..shared)
                    .map(|index| Expert::stacked(&shared_w13, &shared_w2, index, width, hidden))
                    .collect::<candle_core::Result<_>>()?,
                top_k: spec.top_k,
            }))
        } else {
            let (gate, up) = deinterleave(
                &block
                    .pp("w13_dn")
                    .get((2 * intermediate, hidden), "weight")?,
                intermediate,
                hidden,
            )?;
            Inner::Dense {
                expert: Expert {
                    gate,
                    up,
                    down: linear_no_bias(intermediate, hidden, block.pp("w2_md"))?,
                },
                scale: block.get_unchecked("global_scale")?.flatten_all()?,
            }
        };
        Ok(Self {
            inner,
            conv: ResidualConv::load(builder, "mlp_sconv", hidden, spec.kernel, 3)?,
            activation,
        })
    }

    /// The feed-forward over `normed` `[batch, sequence, hidden]`, through
    /// its residual convolution.
    pub(super) fn forward(
        &self,
        normed: &Tensor,
        layer: usize,
        cache: &mut Cache,
    ) -> candle_core::Result<Tensor> {
        let output = match &self.inner {
            Inner::Dense { expert, scale } => expert
                .forward(normed, self.activation)?
                .broadcast_mul(&scale.to_dtype(normed.dtype())?)?,
            Inner::Routed(routed) => routed.forward(normed, self.activation)?,
        };
        self.conv.forward(&output, layer, cache)
    }
}

impl Routed {
    /// The router's weights are read on the host, so they carry no gradient;
    /// the experts' outputs do.
    fn forward(&self, normed: &Tensor, activation: Activation) -> candle_core::Result<Tensor> {
        let (batch, sequence, hidden) = normed.dims3()?;
        let flat = normed.reshape((batch * sequence, hidden))?;
        let logits = self
            .router
            .forward(&flat)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let count = self.experts.len();
        let sigmoid = |logit: f32| 1.0 / (1.0 + (-logit).exp());
        let mut routed: Vec<(Vec<u32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); count];
        let mut shared: Vec<Vec<f32>> = vec![Vec::new(); self.shared.len()];
        for (token, row) in logits.iter().enumerate() {
            let mut order: Vec<usize> = (0..count).collect();
            let ranked: Vec<f32> = (0..count)
                .map(|expert| sigmoid(row[expert]) + self.bias[expert])
                .collect();
            order.sort_by(|left, right| ranked[*right].total_cmp(&ranked[*left]));
            order.truncate(self.top_k);
            let total: f32 = order
                .iter()
                .map(|expert| sigmoid(row[*expert]))
                .sum::<f32>()
                + row[count..]
                    .iter()
                    .map(|logit| sigmoid(*logit))
                    .sum::<f32>();
            for expert in order {
                routed[expert].0.push(token as u32);
                routed[expert]
                    .1
                    .push(sigmoid(row[expert]) / total * self.scale);
            }
            for (index, logit) in row[count..].iter().enumerate() {
                shared[index].push(sigmoid(*logit) / total * self.scale);
            }
        }
        let device = normed.device();
        let mut output = Tensor::zeros((batch * sequence, hidden), normed.dtype(), device)?;
        for (expert, (tokens, weights)) in self.experts.iter().zip(routed) {
            if tokens.is_empty() {
                continue;
            }
            let count = tokens.len();
            let index = Tensor::from_vec(tokens, count, device)?;
            let weights =
                Tensor::from_vec(weights, (count, 1), device)?.to_dtype(normed.dtype())?;
            let produced = expert
                .forward(&flat.index_select(&index, 0)?, activation)?
                .broadcast_mul(&weights)?;
            output = output.index_add(&index, &produced, 0)?;
        }
        for (expert, gammas) in self.shared.iter().zip(shared) {
            let gammas = Tensor::from_vec(gammas, (batch * sequence, 1), device)?
                .to_dtype(normed.dtype())?;
            output = (output + expert.forward(&flat, activation)?.broadcast_mul(&gammas)?)?;
        }
        output.reshape((batch, sequence, hidden))
    }
}
