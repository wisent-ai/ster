//! A mixture-of-experts feed-forward: a router scores every expert for every
//! token, the top `k` run on that token, and their outputs are summed by the
//! router's weights.
//!
//! Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE and GraniteMoE all route this way and
//! differ in tensor names, in whether the chosen weights are renormalised to
//! sum to one, and in whether a shared expert runs on every token beside the
//! routed ones (Qwen2-MoE).

use candle_core::{D, DType, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

use crate::model::{Activation, ExpertLayout, MixtureOfExperts};

#[derive(Debug, Clone)]
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Expert {
    fn forward(&self, input: &Tensor, activation: Activation) -> candle_core::Result<Tensor> {
        let gated = (activation.apply(&self.gate.forward(input)?)? * self.up.forward(input)?)?;
        self.down.forward(&gated)
    }

    fn load(
        hidden: usize,
        intermediate: usize,
        names: [&str; 3],
        builder: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        let [gate, up, down] = names;
        Ok(Self {
            gate: linear_no_bias(hidden, intermediate, builder.pp(gate))?,
            up: linear_no_bias(hidden, intermediate, builder.pp(up))?,
            down: linear_no_bias(intermediate, hidden, builder.pp(down))?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct Experts {
    router: Linear,
    experts: Vec<Expert>,
    /// Qwen2-MoE's expert that every token goes through, and the sigmoid
    /// gate that scales its output.
    shared: Option<(Expert, Linear)>,
    top_k: usize,
    normalize: bool,
    activation: Activation,
}

impl Experts {
    /// `builder` is the layer's.
    pub(super) fn load(
        builder: &VarBuilder<'_>,
        hidden: usize,
        spec: &MixtureOfExperts,
        activation: Activation,
    ) -> candle_core::Result<Self> {
        let count = spec.count;
        let intermediate = spec.intermediate;
        let (router, experts) = match spec.layout {
            ExpertLayout::Mixtral => {
                let block = builder.pp("block_sparse_moe");
                let experts = (0..count)
                    .map(|expert| {
                        Expert::load(
                            hidden,
                            intermediate,
                            ["w1", "w3", "w2"],
                            block.pp("experts").pp(expert.to_string()),
                        )
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("gate"))?, experts)
            }
            ExpertLayout::Qwen => {
                let block = builder.pp("mlp");
                let experts = (0..count)
                    .map(|expert| {
                        Expert::load(
                            hidden,
                            intermediate,
                            ["gate_proj", "up_proj", "down_proj"],
                            block.pp("experts").pp(expert.to_string()),
                        )
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("gate"))?, experts)
            }
            // GraniteMoE stacks every expert into two tensors: `input_linear`
            // `[experts, 2 * intermediate, hidden]`, gate rows then up rows,
            // and `output_linear` `[experts, hidden, intermediate]`. Each
            // expert is a view of its slice, not a copy.
            ExpertLayout::Granite => {
                let block = builder.pp("block_sparse_moe");
                let input = block.get((count, 2 * intermediate, hidden), "input_linear.weight")?;
                let output = block.get((count, hidden, intermediate), "output_linear.weight")?;
                let experts = (0..count)
                    .map(|expert| -> candle_core::Result<Expert> {
                        let rows = input.get(expert)?;
                        Ok(Expert {
                            gate: Linear::new(rows.narrow(0, 0, intermediate)?, None),
                            up: Linear::new(rows.narrow(0, intermediate, intermediate)?, None),
                            down: Linear::new(output.get(expert)?, None),
                        })
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                (linear_no_bias(hidden, count, block.pp("router").pp("layer"))?, experts)
            }
        };
        let shared = match spec.shared_intermediate {
            Some(width) => {
                let block = builder.pp("mlp");
                Some((
                    Expert::load(
                        hidden,
                        width,
                        ["gate_proj", "up_proj", "down_proj"],
                        block.pp("shared_expert"),
                    )?,
                    linear_no_bias(hidden, 1, block.pp("shared_expert_gate"))?,
                ))
            }
            None => None,
        };
        Ok(Self {
            router,
            experts,
            shared,
            top_k: spec.top_k,
            normalize: spec.normalize,
            activation,
        })
    }

    /// Routes every token of `hidden` `[batch, sequence, width]`.
    ///
    /// The choice of experts is a discrete decision read on the host; the
    /// weights stay tensors taken from the router's softmax through a 0/1
    /// mask, so a differentiable pass still carries the router's gradient
    /// back into the residual stream. Every op here — `softmax` composed of
    /// `exp` and `sum`, `index_select`, `index_add`, the mask multiply — has a
    /// backward pass.
    pub(super) fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        let tokens = batch * sequence;
        let flat = hidden.reshape((tokens, width))?;
        let device = flat.device();
        let count = self.experts.len();
        let probabilities =
            candle_nn::ops::softmax(&self.router.forward(&flat)?.to_dtype(DType::F32)?, D::Minus1)?;
        let host = probabilities.to_vec2::<f32>()?;
        let mut mask = vec![0f32; tokens * count];
        let mut routed: Vec<Vec<u32>> = vec![Vec::new(); count];
        for (token, row) in host.iter().enumerate() {
            let mut order: Vec<usize> = (0..count).collect();
            // Highest probability first; equal ones keep expert order, so
            // the choice is deterministic.
            order.sort_by(|left, right| row[*right].total_cmp(&row[*left]));
            for &expert in order.iter().take(self.top_k) {
                mask[token * count + expert] = 1.0;
                routed[expert].push(token as u32);
            }
        }
        let mask = Tensor::from_vec(mask, (tokens, count), device)?;
        let chosen = (probabilities * mask)?;
        let weights = if self.normalize {
            chosen.broadcast_div(&chosen.sum_keepdim(D::Minus1)?)?
        } else {
            chosen
        };
        let mut output = flat.zeros_like()?;
        for (expert, tokens) in routed.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let index = Tensor::new(tokens.as_slice(), device)?;
            let inputs = flat.index_select(&index, 0)?;
            let produced = self.experts[expert].forward(&inputs, self.activation)?;
            let weight = weights
                .index_select(&index, 0)?
                .narrow(1, expert, 1)?
                .to_dtype(produced.dtype())?;
            output = output.index_add(&index, &produced.broadcast_mul(&weight)?, 0)?;
        }
        if let Some((shared, gate)) = &self.shared {
            let gate = gate.forward(&flat)?;
            // sigmoid, composed so it has a backward pass.
            let gate = (gate.neg()?.exp()? + 1.0)?.recip()?;
            output = (output + shared.forward(&flat, self.activation)?.broadcast_mul(&gate)?)?;
        }
        output.reshape((batch, sequence, width))
    }
}
